from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd

from .config import SUPPORTED_LEAGUES
from .exceptions import ValidationError
from .services import FixtureInput, MatchInput
from .time_utils import parse_kickoff_time


MATCH_COLUMNS = [
    "league_code",
    "season",
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
]
FIXTURE_COLUMNS = ["league_code", "season", "match_date", "home_team", "away_team"]


def _read_table(source, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(source)
    raise ValidationError("Upload a .csv or .xlsx file.")


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip().lower().replace(" ", "_") for column in result.columns]
    return result


def parse_match_upload(source, filename: str) -> list[MatchInput]:
    frame = _normalize(_read_table(source, filename))
    missing = [column for column in MATCH_COLUMNS if column not in frame.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValidationError("The upload contains no match rows.")
    allowed = {code for code, _, _ in SUPPORTED_LEAGUES}
    parsed: list[MatchInput] = []
    keys: set[tuple] = set()
    errors: list[str] = []
    for row_number, row in frame.iterrows():
        display_row = row_number + 2
        try:
            league_code = str(row["league_code"]).strip().upper()
            if league_code not in allowed:
                raise ValueError(f"unsupported league_code {league_code!r}")
            parsed_date = pd.to_datetime(row["match_date"], errors="raise").date()
            parsed_time = parse_kickoff_time(row.get("kickoff_time"))
            home_goals = int(row["home_goals"])
            away_goals = int(row["away_goals"])
            if float(row["home_goals"]) != home_goals or float(row["away_goals"]) != away_goals:
                raise ValueError("scores must be whole numbers")
            item = MatchInput(
                league_code=league_code,
                season=str(row["season"]).strip(),
                match_date=parsed_date,
                home_team=str(row["home_team"]).strip(),
                away_team=str(row["away_team"]).strip(),
                home_goals=home_goals,
                away_goals=away_goals,
                kickoff_time=parsed_time,
            )
            key = (league_code, item.kickoff_at, item.home_team.casefold(), item.away_team.casefold())
            if key in keys:
                raise ValueError("duplicate fixture inside upload")
            keys.add(key)
            parsed.append(item)
        except Exception as exc:
            errors.append(f"row {display_row}: {exc}")
    if errors:
        shown = "; ".join(errors[:10])
        suffix = f"; and {len(errors) - 10} more" if len(errors) > 10 else ""
        raise ValidationError(f"Upload validation failed — {shown}{suffix}")
    return parsed


def parse_fixture_upload(source, filename: str) -> list[FixtureInput]:
    frame = _normalize(_read_table(source, filename))
    missing = [column for column in FIXTURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {', '.join(missing)}")
    allowed = {code for code, _, _ in SUPPORTED_LEAGUES}
    fixtures: list[FixtureInput] = []
    for row_number, row in frame.iterrows():
        try:
            league_code = str(row["league_code"]).strip().upper()
            if league_code not in allowed:
                raise ValueError(f"unsupported league_code {league_code!r}")
            fixtures.append(
                FixtureInput(
                    league_code=league_code,
                    season=str(row["season"]).strip() or None,
                    match_date=pd.to_datetime(row["match_date"], errors="raise").date(),
                    home_team=str(row["home_team"]).strip(),
                    away_team=str(row["away_team"]).strip(),
                    kickoff_time=parse_kickoff_time(row.get("kickoff_time")),
                )
            )
        except Exception as exc:
            raise ValidationError(f"Fixture upload row {row_number + 2}: {exc}") from exc
    return fixtures

