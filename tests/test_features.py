from datetime import date, timedelta

from football_predictor.features import generate_training_frame
from football_predictor.models import League
from football_predictor.services import MatchInput, add_completed_matches


def test_features_use_only_prior_matches(session):
    start = date(2026, 1, 1)
    add_completed_matches(
        session,
        [
            MatchInput("ENG-PL", "2025/26", start, "Alpha", "Beta", 4, 0),
            MatchInput("ENG-PL", "2025/26", start + timedelta(days=7), "Alpha", "Beta", 0, 5),
        ],
    )
    league = session.query(League).filter_by(code="ENG-PL").one()
    frame = generate_training_frame(session, league.id)
    assert frame.iloc[0]["home_matches_played"] == 0
    assert frame.iloc[0]["home_points_5"] == 1.0
    assert frame.iloc[1]["home_matches_played"] == 1
    assert frame.iloc[1]["home_points_5"] == 3.0
    assert frame.iloc[1]["home_gf_5"] == 4.0


def test_league_average_excludes_current_result(session):
    start = date(2026, 2, 1)
    add_completed_matches(
        session,
        [
            MatchInput("ESP-LL", "2025/26", start, "One", "Two", 2, 0),
            MatchInput("ESP-LL", "2025/26", start + timedelta(days=1), "Three", "Four", 8, 7),
        ],
    )
    league = session.query(League).filter_by(code="ESP-LL").one()
    frame = generate_training_frame(session, league.id)
    assert frame.iloc[1]["league_home_goals"] == 2.0
    assert frame.iloc[1]["league_away_goals"] == 0.0

