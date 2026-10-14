from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Iterable, Optional

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, aliased

from .elo import rebuild_elo_history
from .exceptions import DuplicateMatchError, ValidationError
from .models import (
    CompletedMatch,
    EloRatingHistory,
    League,
    ModelVersion,
    Prediction,
    PredictionOutcome,
    Season,
    Team,
    UpcomingFixture,
)
from .time_utils import display_kickoff, kickoff_datetime, kickoff_parts, local_now


@dataclass(frozen=True)
class MatchInput:
    league_code: str
    season: str
    match_date: date | datetime
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    kickoff_time: time = time.min

    @property
    def kickoff_at(self) -> datetime:
        return kickoff_datetime(self.match_date, self.kickoff_time)


@dataclass(frozen=True)
class FixtureInput:
    league_code: str
    season: Optional[str]
    match_date: date | datetime
    home_team: str
    away_team: str
    kickoff_time: time = time.min

    @property
    def kickoff_at(self) -> datetime:
        return kickoff_datetime(self.match_date, self.kickoff_time)


@dataclass(frozen=True)
class DeletionSummary:
    deleted_matches: int
    removed_evaluations: int
    deactivated_models: int
    league_codes: tuple[str, ...]


def _clean_name(value: str, field: str) -> str:
    cleaned = " ".join(str(value).strip().split())
    if not cleaned:
        raise ValidationError(f"{field} is required.")
    if len(cleaned) > 128:
        raise ValidationError(f"{field} is too long.")
    return cleaned


def get_league(session: Session, league_code: str) -> League:
    league = session.scalar(select(League).where(League.code == league_code))
    if league is None:
        raise ValidationError(f"Unsupported league code: {league_code}")
    return league


def get_or_create_team(session: Session, league: League, name: str) -> Team:
    cleaned = _clean_name(name, "Team name")
    team = session.scalar(
        select(Team).where(Team.league_id == league.id, func.lower(Team.name) == cleaned.lower())
    )
    if team is None:
        team = Team(league_id=league.id, name=cleaned)
        session.add(team)
        session.flush()
    return team


def get_or_create_season(session: Session, league: League, name: str) -> Season:
    cleaned = _clean_name(name, "Season")
    season = session.scalar(
        select(Season).where(Season.league_id == league.id, Season.name == cleaned)
    )
    if season is None:
        season = Season(league_id=league.id, name=cleaned)
        session.add(season)
        session.flush()
    return season


def _validate_match_input(data: MatchInput) -> None:
    if data.kickoff_at > local_now():
        raise ValidationError("A completed match kickoff cannot be in the future.")
    if data.home_team.strip().casefold() == data.away_team.strip().casefold():
        raise ValidationError("Home and away teams must be different.")
    if isinstance(data.home_goals, bool) or isinstance(data.away_goals, bool):
        raise ValidationError("Scores must be whole numbers.")
    if int(data.home_goals) != data.home_goals or int(data.away_goals) != data.away_goals:
        raise ValidationError("Scores must be whole numbers.")
    if data.home_goals < 0 or data.away_goals < 0:
        raise ValidationError("Scores cannot be negative.")
    if data.home_goals > 30 or data.away_goals > 30:
        raise ValidationError("A score over 30 is probably an input error.")


def add_completed_match(session: Session, data: MatchInput, *, recalculate: bool = True) -> CompletedMatch:
    _validate_match_input(data)
    league = get_league(session, data.league_code)
    season = get_or_create_season(session, league, data.season)
    home = get_or_create_team(session, league, data.home_team)
    away = get_or_create_team(session, league, data.away_team)
    duplicate = session.scalar(
        select(CompletedMatch).where(
            CompletedMatch.league_id == league.id,
            CompletedMatch.match_date == data.kickoff_at,
            CompletedMatch.home_team_id == home.id,
            CompletedMatch.away_team_id == away.id,
        )
    )
    if duplicate:
        raise DuplicateMatchError(
            f"{home.name} vs {away.name} at {display_kickoff(data.kickoff_at)} already exists."
        )
    match = CompletedMatch(
        league_id=league.id,
        season_id=season.id,
        match_date=data.kickoff_at,
        home_team_id=home.id,
        away_team_id=away.id,
        home_goals=int(data.home_goals),
        away_goals=int(data.away_goals),
    )
    session.add(match)
    try:
        session.flush()
    except IntegrityError as exc:
        raise DuplicateMatchError("This completed fixture already exists.") from exc
    if recalculate:
        rebuild_elo_history(session, league.id)
        reconcile_prediction_outcomes(session, league.id)
    return match


def add_completed_matches(session: Session, rows: Iterable[MatchInput]) -> list[CompletedMatch]:
    created: list[CompletedMatch] = []
    affected: set[int] = set()
    for data in rows:
        match = add_completed_match(session, data, recalculate=False)
        created.append(match)
        affected.add(match.league_id)
    for league_id in affected:
        rebuild_elo_history(session, league_id)
        reconcile_prediction_outcomes(session, league_id)
    return created


def update_completed_match(
    session: Session,
    match_id: int,
    *,
    match_date: date | datetime,
    season_name: str,
    home_goals: int,
    away_goals: int,
    kickoff_time: Optional[time] = None,
) -> CompletedMatch:
    match = session.get(CompletedMatch, match_id)
    if match is None:
        raise ValidationError("Match not found.")
    if kickoff_time is None:
        _, kickoff_time = kickoff_parts(match.match_date)
    data = MatchInput(
        league_code=match.league.code,
        season=season_name,
        match_date=match_date,
        home_team=match.home_team.name,
        away_team=match.away_team.name,
        home_goals=home_goals,
        away_goals=away_goals,
        kickoff_time=kickoff_time,
    )
    _validate_match_input(data)
    season = get_or_create_season(session, match.league, season_name)
    match.match_date = data.kickoff_at
    match.season_id = season.id
    match.home_goals = int(home_goals)
    match.away_goals = int(away_goals)
    match.updated_at = datetime.utcnow()
    try:
        session.flush()
    except IntegrityError as exc:
        raise DuplicateMatchError("The edited fixture would duplicate an existing match.") from exc
    rebuild_elo_history(session, match.league_id)
    reconcile_prediction_outcomes(session, match.league_id)
    return match


def add_upcoming_fixture(session: Session, data: FixtureInput) -> UpcomingFixture:
    if data.home_team.strip().casefold() == data.away_team.strip().casefold():
        raise ValidationError("Home and away teams must be different.")
    league = get_league(session, data.league_code)
    home = get_or_create_team(session, league, data.home_team)
    away = get_or_create_team(session, league, data.away_team)
    season_id = get_or_create_season(session, league, data.season).id if data.season else None
    existing = session.scalar(
        select(UpcomingFixture).where(
            UpcomingFixture.league_id == league.id,
            UpcomingFixture.match_date == data.kickoff_at,
            UpcomingFixture.home_team_id == home.id,
            UpcomingFixture.away_team_id == away.id,
        )
    )
    if existing:
        return existing
    fixture = UpcomingFixture(
        league_id=league.id,
        season_id=season_id,
        match_date=data.kickoff_at,
        home_team_id=home.id,
        away_team_id=away.id,
    )
    session.add(fixture)
    session.flush()
    return fixture


def update_upcoming_fixture_kickoff(
    session: Session,
    fixture_id: int,
    *,
    match_date: date | datetime,
    kickoff_time: time,
) -> UpcomingFixture:
    fixture = session.get(UpcomingFixture, fixture_id)
    if fixture is None:
        raise ValidationError("Upcoming fixture not found.")
    kickoff_at = kickoff_datetime(match_date, kickoff_time)
    duplicate = session.scalar(
        select(UpcomingFixture).where(
            UpcomingFixture.id != fixture.id,
            UpcomingFixture.league_id == fixture.league_id,
            UpcomingFixture.match_date == kickoff_at,
            UpcomingFixture.home_team_id == fixture.home_team_id,
            UpcomingFixture.away_team_id == fixture.away_team_id,
        )
    )
    if duplicate is not None:
        raise DuplicateMatchError("That fixture already exists at the selected kickoff time.")
    fixture.match_date = kickoff_at
    try:
        session.flush()
    except IntegrityError as exc:
        raise DuplicateMatchError("That fixture already exists at the selected kickoff time.") from exc
    return fixture


def actual_outcome(home_goals: int, away_goals: int) -> str:
    return "H" if home_goals > away_goals else "A" if away_goals > home_goals else "D"


def reconcile_prediction_outcomes(session: Session, league_id: Optional[int] = None) -> int:
    query = (
        select(Prediction, UpcomingFixture, CompletedMatch)
        .join(UpcomingFixture, Prediction.fixture_id == UpcomingFixture.id)
        .join(
            CompletedMatch,
            and_(
                CompletedMatch.league_id == UpcomingFixture.league_id,
                CompletedMatch.match_date == UpcomingFixture.match_date,
                CompletedMatch.home_team_id == UpcomingFixture.home_team_id,
                CompletedMatch.away_team_id == UpcomingFixture.away_team_id,
            ),
        )
    )
    if league_id is not None:
        query = query.where(UpcomingFixture.league_id == league_id)
    count = 0
    for prediction, fixture, match in session.execute(query).all():
        actual = actual_outcome(match.home_goals, match.away_goals)
        probabilities = {
            "H": prediction.home_win_probability,
            "D": prediction.draw_probability,
            "A": prediction.away_win_probability,
        }
        one_hot = {label: float(label == actual) for label in ("H", "D", "A")}
        brier = sum((probabilities[k] - one_hot[k]) ** 2 for k in probabilities) / 3.0
        event_probability = max(probabilities[actual], 1e-15)
        predicted_btts = prediction.btts_yes_probability >= 0.5
        actual_btts = match.home_goals > 0 and match.away_goals > 0
        predicted_over = prediction.over_25_probability >= 0.5
        actual_over = match.home_goals + match.away_goals > 2
        outcome = session.scalar(
            select(PredictionOutcome).where(PredictionOutcome.prediction_id == prediction.id)
        )
        values = dict(
            completed_match_id=match.id,
            actual_outcome=actual,
            actual_home_goals=match.home_goals,
            actual_away_goals=match.away_goals,
            outcome_correct=prediction.predicted_outcome == actual,
            exact_score_correct=prediction.most_likely_score == f"{match.home_goals}-{match.away_goals}",
            btts_correct=predicted_btts == actual_btts,
            over_25_correct=predicted_over == actual_over,
            brier_score=float(brier),
            log_loss=float(-math.log(event_probability)),
            evaluated_at=datetime.utcnow(),
        )
        if outcome is None:
            outcome = PredictionOutcome(prediction_id=prediction.id, **values)
            session.add(outcome)
        else:
            for key, value in values.items():
                setattr(outcome, key, value)
        fixture.status = "completed"
        count += 1
    session.flush()
    return count


def list_match_rows(
    session: Session,
    *,
    league_code: Optional[str] = None,
    search: str = "",
    limit: int = 2000,
) -> list[dict]:
    home = aliased(Team)
    away = aliased(Team)
    query = (
        select(CompletedMatch, League, Season, home, away)
        .join(League, CompletedMatch.league_id == League.id)
        .join(Season, CompletedMatch.season_id == Season.id)
        .join(home, CompletedMatch.home_team_id == home.id)
        .join(away, CompletedMatch.away_team_id == away.id)
        .order_by(CompletedMatch.match_date.desc(), CompletedMatch.id.desc())
        .limit(limit)
    )
    if league_code:
        query = query.where(League.code == league_code)
    if search.strip():
        pattern = f"%{search.strip()}%"
        query = query.where(or_(home.name.ilike(pattern), away.name.ilike(pattern)))
    return [
        {
            "id": match.id,
            "league": league.name,
            "league_code": league.code,
            "season": season.name,
            "date": kickoff_parts(match.match_date)[0],
            "kickoff_time": kickoff_parts(match.match_date)[1],
            "home_team": home_team.name,
            "away_team": away_team.name,
            "home_goals": match.home_goals,
            "away_goals": match.away_goals,
        }
        for match, league, season, home_team, away_team in session.execute(query).all()
    ]


def find_match_ids(
    session: Session,
    *,
    league_code: Optional[str] = None,
    search: str = "",
) -> list[int]:
    """Return every completed-match ID matching the history filters, without the UI row limit."""
    home = aliased(Team)
    away = aliased(Team)
    query = (
        select(CompletedMatch.id)
        .join(League, CompletedMatch.league_id == League.id)
        .join(home, CompletedMatch.home_team_id == home.id)
        .join(away, CompletedMatch.away_team_id == away.id)
        .order_by(CompletedMatch.match_date.desc(), CompletedMatch.id.desc())
    )
    if league_code:
        query = query.where(League.code == league_code)
    if search.strip():
        pattern = f"%{search.strip()}%"
        query = query.where(or_(home.name.ilike(pattern), away.name.ilike(pattern)))
    return list(session.scalars(query).all())


def delete_completed_matches(session: Session, match_ids: Iterable[int]) -> DeletionSummary:
    """Delete results while preserving prediction snapshots and restoring dependent state."""
    requested_ids = sorted({int(match_id) for match_id in match_ids})
    if not requested_ids:
        raise ValidationError("No completed matches were selected for deletion.")
    matches = list(
        session.scalars(
            select(CompletedMatch)
            .where(CompletedMatch.id.in_(requested_ids))
            .order_by(CompletedMatch.match_date, CompletedMatch.id)
        ).all()
    )
    if not matches:
        raise ValidationError("The selected completed matches no longer exist.")

    existing_ids = [match.id for match in matches]
    league_ids = sorted({match.league_id for match in matches})
    league_codes = tuple(
        session.scalars(select(League.code).where(League.id.in_(league_ids)).order_by(League.code)).all()
    )
    removed_evaluations = session.scalar(
        select(func.count(PredictionOutcome.id)).where(
            PredictionOutcome.completed_match_id.in_(existing_ids)
        )
    ) or 0
    deactivated_models = session.scalar(
        select(func.count(ModelVersion.id)).where(
            ModelVersion.league_id.in_(league_ids), ModelVersion.is_active.is_(True)
        )
    ) or 0

    # A removed result makes the corresponding fixture pending again. Predictions remain immutable.
    for match in matches:
        fixtures = session.scalars(
            select(UpcomingFixture).where(
                UpcomingFixture.league_id == match.league_id,
                UpcomingFixture.match_date == match.match_date,
                UpcomingFixture.home_team_id == match.home_team_id,
                UpcomingFixture.away_team_id == match.away_team_id,
            )
        ).all()
        for fixture in fixtures:
            fixture.status = "scheduled"

    session.execute(
        delete(PredictionOutcome).where(PredictionOutcome.completed_match_id.in_(existing_ids))
    )
    session.execute(delete(EloRatingHistory).where(EloRatingHistory.match_id.in_(existing_ids)))
    session.execute(delete(CompletedMatch).where(CompletedMatch.id.in_(existing_ids)))
    session.execute(
        update(ModelVersion)
        .where(ModelVersion.league_id.in_(league_ids), ModelVersion.is_active.is_(True))
        .values(is_active=False)
    )
    session.flush()
    for league_id in league_ids:
        rebuild_elo_history(session, league_id)
    return DeletionSummary(
        deleted_matches=len(matches),
        removed_evaluations=int(removed_evaluations),
        deactivated_models=int(deactivated_models),
        league_codes=league_codes,
    )

