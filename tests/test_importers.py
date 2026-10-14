from io import StringIO

import pytest

from football_predictor.exceptions import ValidationError
from football_predictor.importers import parse_match_upload


def test_csv_import_validation():
    csv = StringIO(
        "league_code,season,match_date,home_team,away_team,home_goals,away_goals\n"
        "ENG-PL,2025/26,2026-01-01,Alpha,Beta,2,1\n"
    )
    rows = parse_match_upload(csv, "results.csv")
    assert rows[0].home_goals == 2
    assert rows[0].league_code == "ENG-PL"


def test_upload_rejects_duplicate_rows():
    body = (
        "league_code,season,match_date,home_team,away_team,home_goals,away_goals\n"
        "ENG-PL,2025/26,2026-01-01,Alpha,Beta,2,1\n"
        "ENG-PL,2025/26,2026-01-01,Alpha,Beta,1,1\n"
    )
    with pytest.raises(ValidationError, match="duplicate fixture"):
        parse_match_upload(StringIO(body), "results.csv")

