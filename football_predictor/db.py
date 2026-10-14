from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import Engine, create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from .config import SUPPORTED_LEAGUES, get_settings
from .models import Base, League


_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def create_db_engine(database_url: Optional[str] = None) -> Engine:
    url = database_url or get_settings().database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, future=True, connect_args=connect_args)
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


def initialize_database(engine: Optional[Engine] = None) -> None:
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for code, country, name in SUPPORTED_LEAGUES:
            existing = session.scalar(select(League).where(League.code == code))
            if existing is None:
                session.add(League(code=code, country=country, name=name))
        session.commit()


def reset_connections_for_tests() -> None:
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None

