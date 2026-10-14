from datetime import date, timedelta

import pytest
from sqlalchemy import func, select

from football_predictor.exceptions import DuplicateMatchError
from football_predictor.models import EloRatingHistory
from football_predictor.services import MatchInput, add_completed_match, update_completed_match


def test_add_match_prevents_duplicates_and_rebuilds_elo(session):
    item = MatchInput("ENG-PL", "2025/26", date(2026, 8, 1), "Arsenal", "Chelsea", 2, 1)
    match = add_completed_match(session, item)
    assert match.id is not None
    assert session.scalar(select(func.count(EloRatingHistory.id))) == 2
    with pytest.raises(DuplicateMatchError):
        add_completed_match(session, item)


def test_editing_result_recalculates_ratings(session):
    match = add_completed_match(
        session, MatchInput("ENG-PL", "2025/26", date(2026, 8, 1), "Arsenal", "Chelsea", 2, 1)
    )
    first_home_rating = session.scalar(
        select(EloRatingHistory.rating_after).where(EloRatingHistory.team_id == match.home_team_id)
    )
    update_completed_match(
        session,
        match.id,
        match_date=match.match_date,
        season_name="2025/26",
        home_goals=0,
        away_goals=3,
    )
    new_home_rating = session.scalar(
        select(EloRatingHistory.rating_after).where(EloRatingHistory.team_id == match.home_team_id)
    )
    assert new_home_rating < first_home_rating

