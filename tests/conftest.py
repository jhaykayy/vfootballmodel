from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from football_predictor.config import SUPPORTED_LEAGUES
from football_predictor.models import Base, League


@pytest.fixture()
def engine(tmp_path):
    database = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{database}", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all([League(code=code, country=country, name=name) for code, country, name in SUPPORTED_LEAGUES])
        session.commit()
    yield engine
    engine.dispose()


@pytest.fixture()
def session(engine):
    with Session(engine) as session:
        yield session
        session.rollback()

