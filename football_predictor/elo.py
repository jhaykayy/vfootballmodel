from __future__ import annotations

import math
from collections import defaultdict

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .models import CompletedMatch, EloRatingHistory


DEFAULT_ELO = 1500.0
HOME_ADVANTAGE = 60.0
K_FACTOR = 20.0


def expected_home_score(home_rating: float, away_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating - HOME_ADVANTAGE) / 400.0))


def update_ratings(
    home_rating: float, away_rating: float, home_goals: int, away_goals: int
) -> tuple[float, float]:
    if home_goals > away_goals:
        actual = 1.0
    elif home_goals == away_goals:
        actual = 0.5
    else:
        actual = 0.0
    expected = expected_home_score(home_rating, away_rating)
    margin = abs(home_goals - away_goals)
    multiplier = 1.0 if margin <= 1 else math.log1p(margin)
    delta = K_FACTOR * multiplier * (actual - expected)
    return home_rating + delta, away_rating - delta


def rebuild_elo_history(session: Session, league_id: int) -> dict[int, float]:
    session.execute(delete(EloRatingHistory).where(EloRatingHistory.league_id == league_id))
    ratings: dict[int, float] = defaultdict(lambda: DEFAULT_ELO)
    matches = session.scalars(
        select(CompletedMatch)
        .where(CompletedMatch.league_id == league_id)
        .order_by(CompletedMatch.match_date, CompletedMatch.id)
    ).all()
    for match in matches:
        home_before = ratings[match.home_team_id]
        away_before = ratings[match.away_team_id]
        home_after, away_after = update_ratings(
            home_before, away_before, match.home_goals, match.away_goals
        )
        session.add_all(
            [
                EloRatingHistory(
                    league_id=league_id,
                    team_id=match.home_team_id,
                    match_id=match.id,
                    rating_before=home_before,
                    rating_after=home_after,
                    recorded_at=match.match_date.date(),
                ),
                EloRatingHistory(
                    league_id=league_id,
                    team_id=match.away_team_id,
                    match_id=match.id,
                    rating_before=away_before,
                    rating_after=away_after,
                    recorded_at=match.match_date.date(),
                ),
            ]
        )
        ratings[match.home_team_id] = home_after
        ratings[match.away_team_id] = away_after
    session.flush()
    return dict(ratings)

