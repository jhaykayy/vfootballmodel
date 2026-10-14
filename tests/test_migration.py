from datetime import date

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from football_predictor.db import initialize_database
from football_predictor.migration import MatchHistoryConflictError, migrate_match_history
from football_predictor.models import CompletedMatch, EloRatingHistory
from football_predictor.services import MatchInput, add_completed_matches, update_completed_match


def make_engine(tmp_path, name):
    engine = create_engine(f"sqlite:///{tmp_path / name}", future=True)
    initialize_database(engine)
    return engine


def seed_source(engine):
    with Session(engine) as session:
        add_completed_matches(
            session,
            [
                MatchInput("ENG-PL", "2025/26", date(2026, 8, 1), "Arsenal", "Chelsea", 2, 1),
                MatchInput("ENG-PL", "2025/26", date(2026, 8, 8), "Chelsea", "Arsenal", 0, 0),
            ],
        )
        session.commit()


def test_match_history_migration_is_verified_and_repeatable(tmp_path):
    source = make_engine(tmp_path, "source.db")
    target = make_engine(tmp_path, "target.db")
    seed_source(source)

    first = migrate_match_history(source, target)
    second = migrate_match_history(source, target)

    assert first.created == 2
    assert first.skipped == 0
    assert first.target_after == 2
    assert second.created == 0
    assert second.skipped == 2
    with Session(target) as session:
        assert session.scalar(select(func.count(CompletedMatch.id))) == 2
        assert session.scalar(select(func.count(EloRatingHistory.id))) == 4


def test_match_history_migration_rejects_conflicting_duplicate(tmp_path):
    source = make_engine(tmp_path, "source.db")
    target = make_engine(tmp_path, "target.db")
    seed_source(source)
    migrate_match_history(source, target)

    with Session(target) as session:
        match = session.scalar(select(CompletedMatch).order_by(CompletedMatch.id))
        update_completed_match(
            session,
            match.id,
            match_date=match.match_date,
            season_name=match.season.name,
            home_goals=5,
            away_goals=0,
        )
        session.commit()

    with pytest.raises(MatchHistoryConflictError):
        migrate_match_history(source, target)

    with Session(target) as session:
        assert session.scalar(select(func.count(CompletedMatch.id))) == 2
