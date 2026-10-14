from datetime import date, timedelta

from sqlalchemy import func, select

from football_predictor.config import Settings
from football_predictor.models import League, Prediction, PredictionOutcome
from football_predictor.prediction import generate_prediction
from football_predictor.services import (
    FixtureInput,
    MatchInput,
    add_completed_match,
    add_completed_matches,
    add_upcoming_fixture,
)
from football_predictor.training import train_league


def test_training_prediction_and_immutable_history(session, tmp_path):
    teams = ["Alpha", "Beta", "Gamma", "Delta"]
    start = date(2025, 1, 1)
    rows = []
    for index in range(24):
        home = teams[index % 4]
        away = teams[(index + 1 + (index // 4)) % 4]
        if home == away:
            away = teams[(teams.index(home) + 1) % 4]
        rows.append(
            MatchInput(
                "ENG-PL",
                "2024/25",
                start + timedelta(days=index * 4),
                home,
                away,
                (index * 3) % 4,
                (index * 2 + 1) % 3,
            )
        )
    add_completed_matches(session, rows)
    league = session.scalar(select(League).where(League.code == "ENG-PL"))
    settings = Settings(
        database_url="sqlite://",
        model_dir=tmp_path,
        auto_retrain=True,
        min_training_matches=20,
        max_score=9,
        random_state=7,
    )
    version = train_league(session, league.id, settings=settings)
    assert version.is_active
    fixture_day = date(2025, 6, 1)
    fixture = add_upcoming_fixture(
        session, FixtureInput("ENG-PL", "2024/25", fixture_day, "Alpha", "Gamma")
    )
    first = generate_prediction(session, fixture.id)
    second = generate_prediction(session, fixture.id)
    assert first.id != second.id
    assert first.home_win_probability + first.draw_probability + first.away_win_probability == pytest.approx(1.0)
    add_completed_match(
        session,
        MatchInput("ENG-PL", "2024/25", fixture_day, "Alpha", "Gamma", 1, 1),
    )
    assert session.scalar(select(func.count(Prediction.id))) == 2
    assert session.scalar(select(func.count(PredictionOutcome.id))) == 2


import pytest

