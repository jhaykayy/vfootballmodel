from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class League(Base):
    __tablename__ = "leagues"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    country: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(128))
    teams: Mapped[list["Team"]] = relationship(back_populates="league")


class Team(Base):
    __tablename__ = "teams"
    __table_args__ = (UniqueConstraint("league_id", "name", name="uq_team_league_name"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), index=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    league: Mapped[League] = relationship(back_populates="teams")


class Season(Base):
    __tablename__ = "seasons"
    __table_args__ = (UniqueConstraint("league_id", "name", name="uq_season_league_name"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), index=True)
    name: Mapped[str] = mapped_column(String(32))


class CompletedMatch(Base):
    __tablename__ = "completed_matches"
    __table_args__ = (
        UniqueConstraint(
            "league_id", "match_date", "home_team_id", "away_team_id", name="uq_completed_fixture"
        ),
        CheckConstraint("home_team_id != away_team_id", name="ck_different_teams"),
        CheckConstraint("home_goals >= 0 AND away_goals >= 0", name="ck_nonnegative_goals"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), index=True)
    season_id: Mapped[int] = mapped_column(ForeignKey("seasons.id"), index=True)
    match_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    home_goals: Mapped[int] = mapped_column(Integer)
    away_goals: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    home_team: Mapped[Team] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped[Team] = relationship(foreign_keys=[away_team_id])
    league: Mapped[League] = relationship()
    season: Mapped[Season] = relationship()


class UpcomingFixture(Base):
    __tablename__ = "upcoming_fixtures"
    __table_args__ = (
        UniqueConstraint(
            "league_id", "match_date", "home_team_id", "away_team_id", name="uq_upcoming_fixture"
        ),
        CheckConstraint("home_team_id != away_team_id", name="ck_upcoming_different_teams"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), index=True)
    season_id: Mapped[Optional[int]] = mapped_column(ForeignKey("seasons.id"), nullable=True)
    match_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    status: Mapped[str] = mapped_column(String(24), default="scheduled", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    home_team: Mapped[Team] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped[Team] = relationship(foreign_keys=[away_team_id])
    league: Mapped[League] = relationship()


class EloRatingHistory(Base):
    __tablename__ = "elo_rating_history"
    __table_args__ = (UniqueConstraint("match_id", "team_id", name="uq_elo_match_team"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("completed_matches.id", ondelete="CASCADE"), index=True)
    rating_before: Mapped[float] = mapped_column(Float)
    rating_after: Mapped[float] = mapped_column(Float)
    recorded_at: Mapped[date] = mapped_column(Date, index=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), index=True)
    version: Mapped[str] = mapped_column(String(64), unique=True)
    algorithm: Mapped[str] = mapped_column(String(128))
    artifact_path: Mapped[str] = mapped_column(Text)
    training_cutoff: Mapped[datetime] = mapped_column(DateTime)
    training_matches: Mapped[int] = mapped_column(Integer)
    validation_log_loss: Mapped[float] = mapped_column(Float)
    test_log_loss: Mapped[float] = mapped_column(Float)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    promoted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    league: Mapped[League] = relationship()


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"), unique=True, index=True
    )
    payload: Mapped[bytes] = mapped_column(LargeBinary)
    sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SchemaMetadata(Base):
    __tablename__ = "schema_metadata"
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), index=True)
    model_version_id: Mapped[Optional[int]] = mapped_column(ForeignKey("model_versions.id"), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="running")
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    trigger: Mapped[str] = mapped_column(String(32), default="manual")
    promoted: Mapped[bool] = mapped_column(Boolean, default=False)


class ModelMetric(Base):
    __tablename__ = "model_evaluation_metrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    split: Mapped[str] = mapped_column(String(32))
    model_name: Mapped[str] = mapped_column(String(64))
    metric_name: Mapped[str] = mapped_column(String(64))
    metric_value: Mapped[float] = mapped_column(Float)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("upcoming_fixtures.id"), index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    predicted_outcome: Mapped[str] = mapped_column(String(1))
    home_win_probability: Mapped[float] = mapped_column(Float)
    draw_probability: Mapped[float] = mapped_column(Float)
    away_win_probability: Mapped[float] = mapped_column(Float)
    expected_home_goals: Mapped[float] = mapped_column(Float)
    expected_away_goals: Mapped[float] = mapped_column(Float)
    most_likely_score: Mapped[str] = mapped_column(String(16))
    top_scores_json: Mapped[list] = mapped_column(JSON)
    over_15_probability: Mapped[float] = mapped_column(Float)
    over_25_probability: Mapped[float] = mapped_column(Float)
    under_25_probability: Mapped[float] = mapped_column(Float)
    over_35_probability: Mapped[float] = mapped_column(Float)
    btts_yes_probability: Mapped[float] = mapped_column(Float)
    btts_no_probability: Mapped[float] = mapped_column(Float)
    reliability: Mapped[str] = mapped_column(String(24))
    explanation_json: Mapped[dict] = mapped_column(JSON)
    feature_snapshot_json: Mapped[dict] = mapped_column(JSON)
    fixture: Mapped[UpcomingFixture] = relationship()
    model_version: Mapped[ModelVersion] = relationship()


class PredictionOutcome(Base):
    __tablename__ = "prediction_outcomes"
    __table_args__ = (UniqueConstraint("prediction_id", name="uq_prediction_outcome"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), index=True)
    completed_match_id: Mapped[int] = mapped_column(ForeignKey("completed_matches.id"), index=True)
    actual_outcome: Mapped[str] = mapped_column(String(1))
    actual_home_goals: Mapped[int] = mapped_column(Integer)
    actual_away_goals: Mapped[int] = mapped_column(Integer)
    outcome_correct: Mapped[bool] = mapped_column(Boolean)
    exact_score_correct: Mapped[bool] = mapped_column(Boolean)
    btts_correct: Mapped[bool] = mapped_column(Boolean)
    over_25_correct: Mapped[bool] = mapped_column(Boolean)
    brier_score: Mapped[float] = mapped_column(Float)
    log_loss: Mapped[float] = mapped_column(Float)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    prediction: Mapped[Prediction] = relationship()

