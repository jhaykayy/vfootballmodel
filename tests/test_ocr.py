from datetime import date

import pandas as pd
import pytest

from football_predictor.exceptions import ValidationError
from football_predictor.ocr import (
    OCRLine,
    fixture_inputs_from_review,
    match_inputs_from_review,
    match_team_name,
    parse_fixture_lines,
    parse_result_lines,
)


TEAMS = [
    "Arsenal",
    "Chelsea",
    "Manchester City",
    "Manchester United",
    "Liverpool",
    "Tottenham Hotspur",
    "Everton",
]


def line(text: str, confidence: float = 94.0) -> OCRLine:
    return OCRLine(text=text, confidence=confidence)


def test_team_aliases_are_matched_to_database_names():
    matched, confidence, warning = match_team_name("Man Utd", TEAMS)
    assert matched == "Manchester United"
    assert confidence >= 88
    assert warning == ""


def test_parse_multiple_result_layouts_and_date_headings():
    rows = parse_result_lines(
        [
            line("14 October 2026"),
            line("Arsenal 2 - 1 Chelsea"),
            line("Manchester City 3"),
            line("Liverpool 3"),
            line("15 October 2026"),
            line("Tottenham Hotspur"),
            line("1 - 0"),
            line("Everton"),
        ],
        known_teams=TEAMS,
        default_date=date(2026, 10, 1),
        source_image="results.png",
    )
    assert len(rows) == 3
    assert rows[0]["home_team"] == "Arsenal"
    assert (rows[0]["home_goals"], rows[0]["away_goals"]) == (2, 1)
    assert rows[1]["home_team"] == "Manchester City"
    assert (rows[1]["home_goals"], rows[1]["away_goals"]) == (3, 3)
    assert rows[2]["match_date"] == "2026-10-15"
    assert rows[2]["away_team"] == "Everton"


def test_parse_fixture_layouts_and_fallback_date():
    rows = parse_fixture_lines(
        [
            line("16/10/2026"),
            line("Man Utd vs Chelsea"),
            line("Arsenal"),
            line("Liverpool"),
        ],
        known_teams=TEAMS,
        default_date=date(2026, 10, 20),
        source_image="fixtures.jpg",
        day_first=True,
    )
    assert len(rows) == 2
    assert rows[0]["home_team"] == "Manchester United"
    assert rows[0]["match_date"] == "2026-10-16"
    assert rows[1]["home_team"] == "Arsenal"
    assert rows[1]["away_team"] == "Liverpool"


def test_review_conversion_rejects_duplicates_before_database_write():
    frame = pd.DataFrame(
        [
            {
                "include": True,
                "match_date": "2026-10-14",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_goals": 2,
                "away_goals": 1,
            },
            {
                "include": True,
                "match_date": "2026-10-14",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_goals": 1,
                "away_goals": 1,
            },
        ]
    )
    with pytest.raises(ValidationError, match="duplicate fixture"):
        match_inputs_from_review(frame, "ENG-PL", "2026/27")


def test_unselected_rows_are_not_imported():
    frame = pd.DataFrame(
        [
            {
                "include": False,
                "match_date": "bad date",
                "home_team": "",
                "away_team": "",
            },
            {
                "include": True,
                "match_date": "2026-10-20",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
            },
        ]
    )
    fixtures = fixture_inputs_from_review(frame, "ENG-PL", "2026/27")
    assert len(fixtures) == 1
    assert fixtures[0].home_team == "Arsenal"
