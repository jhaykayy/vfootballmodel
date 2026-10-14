from __future__ import annotations

import re
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from typing import Iterable, Optional, Sequence

import pandas as pd
from dateutil import parser as date_parser
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError
from rapidfuzz import fuzz, process

from .exceptions import OCRUnavailableError, ScreenshotParseError, ValidationError
from .services import FixtureInput, MatchInput


Image.MAX_IMAGE_PIXELS = 25_000_000

MAX_IMAGE_BYTES = 12 * 1024 * 1024
MIN_TEAM_MATCH_SCORE = 70.0
HIGH_TEAM_MATCH_SCORE = 88.0

_DATE_NUMERIC = re.compile(r"\b(?:20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]20\d{2})\b")
_DATE_TEXT = re.compile(
    r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)?(?:day)?[,]?\s*"
    r"\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"[,]?\s+20\d{2}\b",
    re.IGNORECASE,
)
_TIME_ONLY = re.compile(r"^\s*\d{1,2}:\d{2}(?:\s*(?:am|pm))?\s*$", re.IGNORECASE)
_SCORE_ONLY = re.compile(r"^\s*(\d{1,2})\s*[-:–—]\s*(\d{1,2})(?:\s*(?:FT|AET|Pens?))?\s*$", re.IGNORECASE)
_TEAM_SCORE = re.compile(r"^\s*(.+?\D)\s+(\d{1,2})(?:\s*(?:FT|AET|Pens?))?\s*$", re.IGNORECASE)
_STATUS_ONLY = re.compile(
    r"^\s*(?:FT|Full\s*Time|AET|Postponed|Cancelled|Today|Tomorrow|Yesterday|Results?|Fixtures?|Matches?)\s*$",
    re.IGNORECASE,
)
_LEAGUE_HEADING = re.compile(
    r"(?:Premier League|La Liga|Ligue 1|Serie A|Bundesliga|England|Spain|France|Italy|Germany)",
    re.IGNORECASE,
)

_COMMON_ALIASES = {
    "man utd": "manchester united",
    "man united": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "newcastle": "newcastle united",
    "west ham": "west ham united",
    "brighton": "brighton and hove albion",
    "psg": "paris saint germain",
    "inter": "inter milan",
    "inter milan": "internazionale",
    "ac milan": "milan",
    "bayern": "bayern munich",
    "dortmund": "borussia dortmund",
    "gladbach": "borussia monchengladbach",
    "atleti": "atletico madrid",
    "athletic": "athletic bilbao",
}


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float
    top: int = 0
    left: int = 0


@dataclass(frozen=True)
class ParsedScreenshot:
    rows: pd.DataFrame
    raw_text: str
    mean_ocr_confidence: float
    warnings: tuple[str, ...]


def ocr_is_available() -> bool:
    return shutil.which("tesseract") is not None


def _preprocess_image(payload: bytes) -> Image.Image:
    if not payload:
        raise ScreenshotParseError("The uploaded image is empty.")
    if len(payload) > MAX_IMAGE_BYTES:
        raise ScreenshotParseError("Each screenshot must be 12 MB or smaller.")
    try:
        with Image.open(BytesIO(payload)) as source:
            if source.width * source.height > Image.MAX_IMAGE_PIXELS:
                raise ScreenshotParseError(
                    "The screenshot is too large. Crop it below 25 megapixels before uploading."
                )
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ScreenshotParseError("The uploaded file is not a safe, readable image.") from exc
    longest = max(image.size)
    scale = 2.0 if longest < 1800 else 1.35 if longest < 2800 else 1.0
    if scale > 1:
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS
        )
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.35)
    return gray.filter(ImageFilter.SHARPEN)


def extract_ocr_lines(payload: bytes) -> list[OCRLine]:
    if not ocr_is_available():
        raise OCRUnavailableError(
            "Tesseract OCR is not installed. On macOS run `brew install tesseract`; "
            "Streamlit Cloud installs it automatically from packages.txt."
        )
    try:
        import pytesseract
        from pytesseract import Output

        image = _preprocess_image(payload)
        data = pytesseract.image_to_data(
            image,
            lang="eng",
            config="--oem 1 --psm 6 -c preserve_interword_spaces=1",
            output_type=Output.DICT,
        )
    except OCRUnavailableError:
        raise
    except Exception as exc:
        raise ScreenshotParseError(f"OCR could not read this screenshot: {exc}") from exc

    grouped: dict[tuple[int, int, int, int], list[tuple[int, str, float, int]]] = {}
    count = len(data.get("text", []))
    for index in range(count):
        word = str(data["text"][index]).strip()
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0
        if not word or confidence < 0:
            continue
        key = tuple(
            int(data[name][index]) for name in ("page_num", "block_num", "par_num", "line_num")
        )
        grouped.setdefault(key, []).append(
            (int(data["left"][index]), word, confidence, int(data["top"][index]))
        )
    lines = []
    for words in grouped.values():
        words.sort(key=lambda item: item[0])
        text = " ".join(item[1] for item in words)
        lines.append(
            OCRLine(
                text=_clean_ocr_text(text),
                confidence=sum(item[2] for item in words) / len(words),
                top=min(item[3] for item in words),
                left=min(item[0] for item in words),
            )
        )
    lines = [line for line in lines if line.text]
    lines.sort(key=lambda line: (line.top, line.left))
    if not lines:
        raise ScreenshotParseError(
            "No readable text was found. Try a sharper screenshot cropped around the match list."
        )
    return lines


def _clean_ocr_text(value: str) -> str:
    value = value.replace("|", " ").replace("—", "-").replace("–", "-")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" •·")


def _normalize_team(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = value.casefold().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    words = [word for word in value.split() if word not in {"fc", "cf", "afc", "sc"}]
    normalized = " ".join(words)
    while normalized in _COMMON_ALIASES and _COMMON_ALIASES[normalized] != normalized:
        normalized = _COMMON_ALIASES[normalized]
    return normalized


def match_team_name(raw_name: str, known_teams: Sequence[str]) -> tuple[str, float, str]:
    cleaned = _clean_ocr_text(re.sub(r"\b(?:FT|AET|Pens?)\b", "", raw_name, flags=re.IGNORECASE))
    cleaned = re.sub(r"^[\d\W]+|[\d\W]+$", "", cleaned).strip()
    if not cleaned:
        return "", 0.0, "team name is empty"
    if not known_teams:
        return cleaned, 50.0, "no existing league teams were available for automatic matching"
    normalized_choices: dict[str, str] = {}
    for team in known_teams:
        normalized_choices[_normalize_team(team)] = team
    query = _normalize_team(cleaned)
    match = process.extractOne(query, list(normalized_choices), scorer=fuzz.WRatio)
    if match is None:
        return cleaned, 0.0, "team was not recognized"
    normalized_match, score, _ = match
    selected = normalized_choices[normalized_match]
    if score < MIN_TEAM_MATCH_SCORE:
        return cleaned, float(score), f"team was not confidently matched (best suggestion: {selected})"
    warning = "" if score >= HIGH_TEAM_MATCH_SCORE else f"verify team match: {cleaned} → {selected}"
    return selected, float(score), warning


def _date_in_line(text: str, *, day_first: bool) -> Optional[date]:
    candidate_match = _DATE_NUMERIC.search(text) or _DATE_TEXT.search(text)
    if not candidate_match:
        return None
    candidate = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", candidate_match.group(0), flags=re.IGNORECASE)
    try:
        parsed = date_parser.parse(candidate, dayfirst=day_first, fuzzy=False).date()
    except (ValueError, OverflowError):
        return None
    if not (2000 <= parsed.year <= date.today().year + 3):
        return None
    return parsed


def _line_dates(lines: Sequence[OCRLine], default_date: date, day_first: bool) -> list[date]:
    current = default_date
    result = []
    for line in lines:
        detected = _date_in_line(line.text, day_first=day_first)
        if detected:
            current = detected
        result.append(current)
    return result


def _is_metadata_line(text: str) -> bool:
    return bool(
        not text
        or _STATUS_ONLY.fullmatch(text)
        or _TIME_ONLY.fullmatch(text)
        or _DATE_NUMERIC.fullmatch(text)
        or _DATE_TEXT.fullmatch(text)
        or (_LEAGUE_HEADING.fullmatch(text) and not any(char.isdigit() for char in text))
    )


def _valid_score(home_goals: int, away_goals: int) -> bool:
    return 0 <= home_goals <= 30 and 0 <= away_goals <= 30


def _build_row(
    *,
    row_date: date,
    raw_home: str,
    raw_away: str,
    known_teams: Sequence[str],
    ocr_confidence: float,
    source_image: str,
    source_text: str,
    position: int,
    home_goals: Optional[int] = None,
    away_goals: Optional[int] = None,
) -> dict:
    home, home_score, home_warning = match_team_name(raw_home, known_teams)
    away, away_score, away_warning = match_team_name(raw_away, known_teams)
    warnings = [warning for warning in (home_warning, away_warning) if warning]
    if home and away and home.casefold() == away.casefold():
        warnings.append("home and away resolved to the same team")
    confidence = round(min(float(ocr_confidence), home_score, away_score))
    if confidence < 70:
        warnings.append("low-confidence OCR row")
    row = {
        "include": True,
        "match_date": row_date.isoformat(),
        "home_team": home,
        "away_team": away,
        "confidence": int(max(0, min(100, confidence))),
        "review": "; ".join(dict.fromkeys(warnings)),
        "source_image": source_image,
        "source_text": source_text,
        "_position": position,
    }
    if home_goals is not None and away_goals is not None:
        row["home_goals"] = home_goals
        row["away_goals"] = away_goals
    return row


def _combined_result(text: str) -> Optional[tuple[str, int, int, str]]:
    patterns = (
        re.compile(r"^\s*(.+?\D)\s+(\d{1,2})\s*[-:–—]\s*(\d{1,2})\s+(.+?\D)\s*$"),
        re.compile(r"^\s*(.+?\D)\s+(\d{1,2})\s+(\d{1,2})\s+(.+?\D)\s*$"),
        re.compile(r"^\s*(.+?\D)\s+(\d{1,2})\s+(.+?\D)\s+(\d{1,2})\s*$"),
    )
    for index, pattern in enumerate(patterns):
        match = pattern.match(text)
        if not match:
            continue
        if index < 2:
            home, home_goals, away_goals, away = match.groups()
        else:
            home, home_goals, away, away_goals = match.groups()
        scores = int(home_goals), int(away_goals)
        if _valid_score(*scores):
            return home.strip(), scores[0], scores[1], away.strip()
    return None


def parse_result_lines(
    lines: Sequence[OCRLine],
    *,
    known_teams: Sequence[str],
    default_date: date,
    source_image: str,
    day_first: bool = True,
) -> list[dict]:
    dates = _line_dates(lines, default_date, day_first)
    rows: list[dict] = []
    used: set[int] = set()

    for index, line in enumerate(lines):
        combined = _combined_result(line.text)
        if combined:
            home, home_goals, away_goals, away = combined
            rows.append(
                _build_row(
                    row_date=dates[index],
                    raw_home=home,
                    raw_away=away,
                    known_teams=known_teams,
                    ocr_confidence=line.confidence,
                    source_image=source_image,
                    source_text=line.text,
                    position=index,
                    home_goals=home_goals,
                    away_goals=away_goals,
                )
            )
            used.add(index)

    for index, line in enumerate(lines):
        if index in used:
            continue
        score_match = _SCORE_ONLY.fullmatch(line.text)
        if not score_match:
            continue
        previous = next(
            (position for position in range(index - 1, -1, -1) if position not in used and not _is_metadata_line(lines[position].text)),
            None,
        )
        following = next(
            (position for position in range(index + 1, len(lines)) if position not in used and not _is_metadata_line(lines[position].text)),
            None,
        )
        if previous is None or following is None:
            continue
        home_goals, away_goals = map(int, score_match.groups())
        if not _valid_score(home_goals, away_goals):
            continue
        rows.append(
            _build_row(
                row_date=dates[index],
                raw_home=lines[previous].text,
                raw_away=lines[following].text,
                known_teams=known_teams,
                ocr_confidence=min(line.confidence, lines[previous].confidence, lines[following].confidence),
                source_image=source_image,
                source_text=" | ".join((lines[previous].text, line.text, lines[following].text)),
                position=previous,
                home_goals=home_goals,
                away_goals=away_goals,
            )
        )
        used.update((previous, index, following))

    candidates = []
    for index, line in enumerate(lines):
        if index in used or _is_metadata_line(line.text):
            continue
        match = _TEAM_SCORE.fullmatch(line.text)
        if match:
            team, score = match.groups()
            candidates.append((index, team.strip(), int(score)))
    for first, second in zip(candidates[0::2], candidates[1::2]):
        first_index, home, home_goals = first
        second_index, away, away_goals = second
        if second_index - first_index > 3 or not _valid_score(home_goals, away_goals):
            continue
        rows.append(
            _build_row(
                row_date=dates[first_index],
                raw_home=home,
                raw_away=away,
                known_teams=known_teams,
                ocr_confidence=min(lines[first_index].confidence, lines[second_index].confidence),
                source_image=source_image,
                source_text=f"{lines[first_index].text} | {lines[second_index].text}",
                position=first_index,
                home_goals=home_goals,
                away_goals=away_goals,
            )
        )
    return _deduplicate_rows(rows, results=True)


def _split_fixture_line(text: str) -> Optional[tuple[str, str]]:
    versus = re.match(r"^\s*(.+?)\s+(?:vs?\.?|versus)\s+(.+?)\s*$", text, re.IGNORECASE)
    if versus:
        return versus.group(1), versus.group(2)
    hyphen = re.match(r"^\s*(.+?\D)\s+[-–—]\s+(.+?\D)\s*$", text)
    if hyphen:
        return hyphen.group(1), hyphen.group(2)
    return None


def parse_fixture_lines(
    lines: Sequence[OCRLine],
    *,
    known_teams: Sequence[str],
    default_date: date,
    source_image: str,
    day_first: bool = True,
) -> list[dict]:
    dates = _line_dates(lines, default_date, day_first)
    rows: list[dict] = []
    used: set[int] = set()
    for index, line in enumerate(lines):
        split = _split_fixture_line(line.text)
        if not split:
            continue
        rows.append(
            _build_row(
                row_date=dates[index],
                raw_home=split[0],
                raw_away=split[1],
                known_teams=known_teams,
                ocr_confidence=line.confidence,
                source_image=source_image,
                source_text=line.text,
                position=index,
            )
        )
        used.add(index)

    team_lines: list[tuple[int, str, float]] = []
    for index, line in enumerate(lines):
        if index in used or _is_metadata_line(line.text) or any(char.isdigit() for char in line.text):
            continue
        matched, score, _ = match_team_name(line.text, known_teams)
        if matched and (not known_teams or score >= MIN_TEAM_MATCH_SCORE):
            team_lines.append((index, line.text, score))
    for first, second in zip(team_lines[0::2], team_lines[1::2]):
        first_index, home, _ = first
        second_index, away, _ = second
        if second_index - first_index > 3:
            continue
        rows.append(
            _build_row(
                row_date=dates[first_index],
                raw_home=home,
                raw_away=away,
                known_teams=known_teams,
                ocr_confidence=min(lines[first_index].confidence, lines[second_index].confidence),
                source_image=source_image,
                source_text=f"{lines[first_index].text} | {lines[second_index].text}",
                position=first_index,
            )
        )
    return _deduplicate_rows(rows, results=False)


def _deduplicate_rows(rows: list[dict], *, results: bool) -> list[dict]:
    deduplicated = []
    seen = set()
    for original in sorted(rows, key=lambda item: item.get("_position", 0)):
        row = dict(original)
        row.pop("_position", None)
        key = (row["match_date"], row["home_team"].casefold(), row["away_team"].casefold())
        if results:
            key += (row["home_goals"], row["away_goals"])
        if key not in seen:
            deduplicated.append(row)
            seen.add(key)
    return deduplicated


def parse_screenshot(
    payload: bytes,
    *,
    mode: str,
    known_teams: Sequence[str],
    default_date: date,
    source_image: str,
    day_first: bool = True,
) -> ParsedScreenshot:
    lines = extract_ocr_lines(payload)
    if mode == "results":
        rows = parse_result_lines(
            lines,
            known_teams=known_teams,
            default_date=default_date,
            source_image=source_image,
            day_first=day_first,
        )
    elif mode == "fixtures":
        rows = parse_fixture_lines(
            lines,
            known_teams=known_teams,
            default_date=default_date,
            source_image=source_image,
            day_first=day_first,
        )
    else:
        raise ValueError("mode must be 'results' or 'fixtures'")
    raw_text = "\n".join(line.text for line in lines)
    mean_confidence = sum(line.confidence for line in lines) / len(lines)
    warnings = []
    if not rows:
        warnings.append(
            "No match rows were recognized. Crop the image to the match list or enter rows manually in the preview."
        )
    if mean_confidence < 65:
        warnings.append("Overall OCR confidence is low; verify every extracted field carefully.")
    return ParsedScreenshot(pd.DataFrame(rows), raw_text, mean_confidence, tuple(warnings))


def match_inputs_from_review(frame: pd.DataFrame, league_code: str, season: str) -> list[MatchInput]:
    if not season.strip():
        raise ValidationError("Season is required before confirming screenshot results.")
    selected = frame[frame["include"].fillna(False).astype(bool)]
    if selected.empty:
        raise ValidationError("Select at least one extracted result to import.")
    rows = []
    errors = []
    seen = set()
    for index, row in selected.iterrows():
        try:
            item = MatchInput(
                league_code=league_code,
                season=season.strip(),
                match_date=pd.to_datetime(row["match_date"], errors="raise").date(),
                home_team=str(row["home_team"]).strip(),
                away_team=str(row["away_team"]).strip(),
                home_goals=int(row["home_goals"]),
                away_goals=int(row["away_goals"]),
            )
            if not item.home_team or not item.away_team:
                raise ValueError("both team names are required")
            if item.home_team.casefold() == item.away_team.casefold():
                raise ValueError("home and away teams must be different")
            if float(row["home_goals"]) != item.home_goals or float(row["away_goals"]) != item.away_goals:
                raise ValueError("scores must be whole numbers")
            if not _valid_score(item.home_goals, item.away_goals):
                raise ValueError("scores must be between 0 and 30")
            key = (item.match_date, item.home_team.casefold(), item.away_team.casefold())
            if key in seen:
                raise ValueError("duplicate fixture in confirmation table")
            seen.add(key)
            rows.append(item)
        except Exception as exc:
            errors.append(f"row {index + 1}: {exc}")
    if errors:
        raise ValidationError("Screenshot confirmation failed — " + "; ".join(errors[:10]))
    return rows


def fixture_inputs_from_review(
    frame: pd.DataFrame, league_code: str, season: Optional[str]
) -> list[FixtureInput]:
    selected = frame[frame["include"].fillna(False).astype(bool)]
    if selected.empty:
        raise ValidationError("Select at least one extracted fixture to import.")
    rows = []
    errors = []
    seen = set()
    for index, row in selected.iterrows():
        try:
            item = FixtureInput(
                league_code=league_code,
                season=season.strip() if season and season.strip() else None,
                match_date=pd.to_datetime(row["match_date"], errors="raise").date(),
                home_team=str(row["home_team"]).strip(),
                away_team=str(row["away_team"]).strip(),
            )
            if not item.home_team or not item.away_team:
                raise ValueError("both team names are required")
            if item.home_team.casefold() == item.away_team.casefold():
                raise ValueError("home and away teams must be different")
            if item.match_date < date.today():
                raise ValueError("an upcoming fixture date cannot be in the past")
            key = (item.match_date, item.home_team.casefold(), item.away_team.casefold())
            if key in seen:
                raise ValueError("duplicate fixture in confirmation table")
            seen.add(key)
            rows.append(item)
        except Exception as exc:
            errors.append(f"row {index + 1}: {exc}")
    if errors:
        raise ValidationError("Screenshot confirmation failed — " + "; ".join(errors[:10]))
    return rows
