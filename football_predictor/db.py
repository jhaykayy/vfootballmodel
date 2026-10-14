from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from datetime import datetime, time

from sqlalchemy import DateTime, Engine, create_engine, event, func, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from .config import SUPPORTED_LEAGUES, get_settings
from .models import (
    Base,
    CompletedMatch,
    League,
    ModelVersion,
    SchemaMetadata,
    Season,
    UpcomingFixture,
)
from .time_utils import kickoff_datetime, parse_kickoff_time


_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def normalize_database_url(database_url: str) -> str:
    """Use Psycopg 3 for PostgreSQL URLs supplied by hosting providers."""
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


def create_db_engine(database_url: Optional[str] = None) -> Engine:
    url = normalize_database_url(database_url or get_settings().database_url)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine_options = {
        "future": True,
        "connect_args": connect_args,
    }
    if url.startswith("postgresql"):
        engine_options.update(
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=3,
            max_overflow=2,
        )
    engine = create_engine(url, **engine_options)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _enable_sqlite_constraints(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return engine


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_db_engine()
    return _engine


def get_session_factory() -> sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _upgrade_postgresql_datetime_columns(engine: Engine) -> None:
    if engine.dialect.name != "postgresql":
        return
    inspector = inspect(engine)
    targets = (
        ("completed_matches", "match_date"),
        ("upcoming_fixtures", "match_date"),
        ("model_versions", "training_cutoff"),
    )
    statements = []
    for table_name, column_name in targets:
        if not inspector.has_table(table_name):
            continue
        column = next(
            (item for item in inspector.get_columns(table_name) if item["name"] == column_name),
            None,
        )
        if column is not None and not isinstance(column["type"], DateTime):
            statements.append(
                f'ALTER TABLE "{table_name}" ALTER COLUMN "{column_name}" '
                f'TYPE TIMESTAMP WITHOUT TIME ZONE USING "{column_name}"::timestamp'
            )
    if statements:
        with engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))


def _backfill_kickoff_timestamps(session: Session) -> None:
    marker_key = "kickoff-timestamps-v1"
    if session.get(SchemaMetadata, marker_key) is not None:
        return
    seasons = {item.id: item.name for item in session.scalars(select(Season)).all()}

    def inferred_time(season_id: int | None) -> time:
        if season_id is None:
            return time.min
        try:
            return parse_kickoff_time(seasons.get(season_id))
        except ValueError:
            return time.min

    affected_leagues: set[int] = set()
    for match in session.scalars(select(CompletedMatch)).all():
        stamp = kickoff_datetime(match.match_date)
        guessed = inferred_time(match.season_id)
        if stamp.time() == time.min and guessed != time.min:
            match.match_date = datetime.combine(stamp.date(), guessed)
        else:
            match.match_date = stamp
        affected_leagues.add(match.league_id)
    for fixture in session.scalars(select(UpcomingFixture)).all():
        stamp = kickoff_datetime(fixture.match_date)
        guessed = inferred_time(fixture.season_id)
        fixture.match_date = (
            datetime.combine(stamp.date(), guessed)
            if stamp.time() == time.min and guessed != time.min
            else stamp
        )
    for version in session.scalars(select(ModelVersion)).all():
        latest = session.scalar(
            select(func.max(CompletedMatch.match_date)).where(
                CompletedMatch.league_id == version.league_id
            )
        )
        version.training_cutoff = kickoff_datetime(latest or version.training_cutoff)
        # Earlier artifacts were trained with date-only ordering and cannot be reused safely.
        version.is_active = False
    session.flush()
    if affected_leagues:
        from .elo import rebuild_elo_history

        for league_id in affected_leagues:
            rebuild_elo_history(session, league_id)
    session.add(SchemaMetadata(key=marker_key, value="completed"))


def initialize_database(engine: Optional[Engine] = None) -> None:
    engine = engine or get_engine()
    _upgrade_postgresql_datetime_columns(engine)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for code, country, name in SUPPORTED_LEAGUES:
            existing = session.scalar(select(League).where(League.code == code))
            if existing is None:
                session.add(League(code=code, country=country, name=name))
        session.flush()
        _backfill_kickoff_timestamps(session)
        session.commit()


def reset_connections_for_tests() -> None:
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None

