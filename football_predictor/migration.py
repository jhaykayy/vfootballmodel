from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, aliased, joinedload

from .db import initialize_database
from .elo import rebuild_elo_history
from .models import CompletedMatch, League, Team
from .services import MatchInput, add_completed_match, reconcile_prediction_outcomes


@dataclass(frozen=True)
class MatchHistoryMigrationSummary:
    source_matches: int
    target_before: int
    created: int
    skipped: int
    target_after: int


class MatchHistoryConflictError(RuntimeError):
    """Raised when the target has the same fixture with different result data."""


def read_match_history(source_engine: Engine) -> list[MatchInput]:
    with Session(source_engine) as session:
        matches = session.scalars(
            select(CompletedMatch)
            .options(
                joinedload(CompletedMatch.league),
                joinedload(CompletedMatch.season),
                joinedload(CompletedMatch.home_team),
                joinedload(CompletedMatch.away_team),
            )
            .order_by(CompletedMatch.match_date, CompletedMatch.id)
        ).all()
        return [
            MatchInput(
                league_code=match.league.code,
                season=match.season.name,
                match_date=match.match_date,
                home_team=match.home_team.name,
                away_team=match.away_team.name,
                home_goals=match.home_goals,
                away_goals=match.away_goals,
            )
            for match in matches
        ]


def _find_existing_match(session: Session, item: MatchInput) -> CompletedMatch | None:
    home = aliased(Team)
    away = aliased(Team)
    return session.scalar(
        select(CompletedMatch)
        .join(League, CompletedMatch.league_id == League.id)
        .join(home, CompletedMatch.home_team_id == home.id)
        .join(away, CompletedMatch.away_team_id == away.id)
        .where(
            League.code == item.league_code,
            CompletedMatch.match_date == item.match_date,
            func.lower(home.name) == item.home_team.lower(),
            func.lower(away.name) == item.away_team.lower(),
        )
        .options(joinedload(CompletedMatch.season))
    )


def migrate_match_history(source_engine: Engine, target_engine: Engine) -> MatchHistoryMigrationSummary:
    """Copy completed matches without changing or deleting the source database."""
    if source_engine.url.render_as_string(hide_password=False) == target_engine.url.render_as_string(
        hide_password=False
    ):
        raise ValueError("Source and target databases must be different.")

    rows = read_match_history(source_engine)
    initialize_database(target_engine)

    with Session(target_engine) as session:
        target_before = int(session.scalar(select(func.count(CompletedMatch.id))) or 0)
        created = 0
        skipped = 0
        affected_leagues: set[int] = set()

        for item in rows:
            existing = _find_existing_match(session, item)
            if existing is not None:
                same_values = (
                    existing.season.name == item.season
                    and existing.home_goals == item.home_goals
                    and existing.away_goals == item.away_goals
                )
                if not same_values:
                    raise MatchHistoryConflictError(
                        "Conflicting target match: "
                        f"{item.match_date.isoformat()} {item.home_team} vs {item.away_team}. "
                        "Nothing from this migration was committed."
                    )
                skipped += 1
                continue

            match = add_completed_match(session, item, recalculate=False)
            affected_leagues.add(match.league_id)
            created += 1

        for league_id in affected_leagues:
            rebuild_elo_history(session, league_id)
            reconcile_prediction_outcomes(session, league_id)

        target_after = int(session.scalar(select(func.count(CompletedMatch.id))) or 0)

        for item in rows:
            copied = _find_existing_match(session, item)
            if copied is None or (
                copied.season.name != item.season
                or copied.home_goals != item.home_goals
                or copied.away_goals != item.away_goals
            ):
                raise RuntimeError(
                    "Post-migration verification failed for "
                    f"{item.match_date.isoformat()} {item.home_team} vs {item.away_team}."
                )
        session.commit()

    return MatchHistoryMigrationSummary(
        source_matches=len(rows),
        target_before=target_before,
        created=created,
        skipped=skipped,
        target_after=target_after,
    )
