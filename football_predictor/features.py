from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from .elo import DEFAULT_ELO, update_ratings
from .models import CompletedMatch


FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_matches_played",
    "away_matches_played",
    "home_points_5",
    "away_points_5",
    "home_points_10",
    "away_points_10",
    "home_gf_5",
    "away_gf_5",
    "home_ga_5",
    "away_ga_5",
    "home_gf_10",
    "away_gf_10",
    "home_ga_10",
    "away_ga_10",
    "home_venue_points_10",
    "away_venue_points_10",
    "home_venue_gf_10",
    "away_venue_gf_10",
    "home_venue_ga_10",
    "away_venue_ga_10",
    "home_win_streak",
    "away_win_streak",
    "home_loss_streak",
    "away_loss_streak",
    "home_rest_days",
    "away_rest_days",
    "home_points_vs_strong_10",
    "away_points_vs_strong_10",
    "home_points_vs_weak_10",
    "away_points_vs_weak_10",
    "h2h_count",
    "h2h_home_points",
    "h2h_home_gf",
    "h2h_home_ga",
    "league_home_goals",
    "league_away_goals",
    "league_home_advantage",
]


@dataclass
class Observation:
    played_on: date
    goals_for: int
    goals_against: int
    points: int
    venue: str
    opponent_category: str


@dataclass
class TeamState:
    elo: float = DEFAULT_ELO
    observations: list[Observation] = field(default_factory=list)
    last_match: Optional[date] = None


def _mean(items: list[Observation], attr: str, fallback: float) -> float:
    return float(np.mean([getattr(item, attr) for item in items])) if items else fallback


def _recent(state: TeamState, n: int, venue: Optional[str] = None) -> list[Observation]:
    items = state.observations
    if venue is not None:
        items = [item for item in items if item.venue == venue]
    return items[-n:]


def _streak(state: TeamState, points_value: int) -> int:
    streak = 0
    for observation in reversed(state.observations):
        if observation.points != points_value:
            break
        streak += 1
    return streak


def _points_against(state: TeamState, category: str) -> float:
    matches = [item for item in state.observations[-10:] if item.opponent_category == category]
    return _mean(matches, "points", 1.0)


class ChronologicalFeatureBuilder:
    """Builds every row before applying that match's result."""

    def __init__(self) -> None:
        self.states: dict[int, TeamState] = defaultdict(TeamState)
        self.total_home_goals = 0
        self.total_away_goals = 0
        self.match_count = 0
        self.h2h: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)

    @property
    def league_home_goals(self) -> float:
        return self.total_home_goals / self.match_count if self.match_count else 1.45

    @property
    def league_away_goals(self) -> float:
        return self.total_away_goals / self.match_count if self.match_count else 1.15

    def build_row(self, home_team_id: int, away_team_id: int, played_on: date) -> dict[str, float]:
        home = self.states[home_team_id]
        away = self.states[away_team_id]
        league_home = self.league_home_goals
        league_away = self.league_away_goals
        home_5, away_5 = _recent(home, 5), _recent(away, 5)
        home_10, away_10 = _recent(home, 10), _recent(away, 10)
        home_venue = _recent(home, 10, "H")
        away_venue = _recent(away, 10, "A")
        key = tuple(sorted((home_team_id, away_team_id)))
        history = self.h2h[key][-5:]
        # H2H is stored in sorted-team perspective and flipped when needed.
        h2h_home_goals: list[int] = []
        h2h_away_goals: list[int] = []
        for first_goals, second_goals in history:
            if home_team_id == key[0]:
                h2h_home_goals.append(first_goals)
                h2h_away_goals.append(second_goals)
            else:
                h2h_home_goals.append(second_goals)
                h2h_away_goals.append(first_goals)
        h2h_points = [3 if gf > ga else 1 if gf == ga else 0 for gf, ga in zip(h2h_home_goals, h2h_away_goals)]
        return {
            "home_elo": home.elo,
            "away_elo": away.elo,
            "elo_diff": home.elo - away.elo,
            "home_matches_played": len(home.observations),
            "away_matches_played": len(away.observations),
            "home_points_5": _mean(home_5, "points", 1.0),
            "away_points_5": _mean(away_5, "points", 1.0),
            "home_points_10": _mean(home_10, "points", 1.0),
            "away_points_10": _mean(away_10, "points", 1.0),
            "home_gf_5": _mean(home_5, "goals_for", league_home),
            "away_gf_5": _mean(away_5, "goals_for", league_away),
            "home_ga_5": _mean(home_5, "goals_against", league_away),
            "away_ga_5": _mean(away_5, "goals_against", league_home),
            "home_gf_10": _mean(home_10, "goals_for", league_home),
            "away_gf_10": _mean(away_10, "goals_for", league_away),
            "home_ga_10": _mean(home_10, "goals_against", league_away),
            "away_ga_10": _mean(away_10, "goals_against", league_home),
            "home_venue_points_10": _mean(home_venue, "points", 1.0),
            "away_venue_points_10": _mean(away_venue, "points", 1.0),
            "home_venue_gf_10": _mean(home_venue, "goals_for", league_home),
            "away_venue_gf_10": _mean(away_venue, "goals_for", league_away),
            "home_venue_ga_10": _mean(home_venue, "goals_against", league_away),
            "away_venue_ga_10": _mean(away_venue, "goals_against", league_home),
            "home_win_streak": _streak(home, 3),
            "away_win_streak": _streak(away, 3),
            "home_loss_streak": _streak(home, 0),
            "away_loss_streak": _streak(away, 0),
            "home_rest_days": min((played_on - home.last_match).days, 30) if home.last_match else 7,
            "away_rest_days": min((played_on - away.last_match).days, 30) if away.last_match else 7,
            "home_points_vs_strong_10": _points_against(home, "strong"),
            "away_points_vs_strong_10": _points_against(away, "strong"),
            "home_points_vs_weak_10": _points_against(home, "weak"),
            "away_points_vs_weak_10": _points_against(away, "weak"),
            "h2h_count": len(history),
            "h2h_home_points": float(np.mean(h2h_points)) if h2h_points else 1.0,
            "h2h_home_gf": float(np.mean(h2h_home_goals)) if h2h_home_goals else league_home,
            "h2h_home_ga": float(np.mean(h2h_away_goals)) if h2h_away_goals else league_away,
            "league_home_goals": league_home,
            "league_away_goals": league_away,
            "league_home_advantage": league_home - league_away,
        }

    def apply_result(
        self,
        home_team_id: int,
        away_team_id: int,
        played_on: date,
        home_goals: int,
        away_goals: int,
    ) -> None:
        home = self.states[home_team_id]
        away = self.states[away_team_id]
        home_before, away_before = home.elo, away.elo
        if away_before > home_before + 75:
            home_category = "strong"
        elif away_before < home_before - 75:
            home_category = "weak"
        else:
            home_category = "peer"
        if home_before > away_before + 75:
            away_category = "strong"
        elif home_before < away_before - 75:
            away_category = "weak"
        else:
            away_category = "peer"
        home_points = 3 if home_goals > away_goals else 1 if home_goals == away_goals else 0
        away_points = 3 if away_goals > home_goals else 1 if away_goals == home_goals else 0
        home.observations.append(
            Observation(played_on, home_goals, away_goals, home_points, "H", home_category)
        )
        away.observations.append(
            Observation(played_on, away_goals, home_goals, away_points, "A", away_category)
        )
        home.last_match = played_on
        away.last_match = played_on
        home.elo, away.elo = update_ratings(home_before, away_before, home_goals, away_goals)
        key = tuple(sorted((home_team_id, away_team_id)))
        if home_team_id == key[0]:
            self.h2h[key].append((home_goals, away_goals))
        else:
            self.h2h[key].append((away_goals, home_goals))
        self.total_home_goals += home_goals
        self.total_away_goals += away_goals
        self.match_count += 1


def _ordered_matches(session: Session, league_id: int, before: Optional[date] = None) -> list[CompletedMatch]:
    query = select(CompletedMatch).where(CompletedMatch.league_id == league_id)
    if before is not None:
        query = query.where(CompletedMatch.match_date < before)
    return list(session.scalars(query.order_by(CompletedMatch.match_date, CompletedMatch.id)).all())


def generate_training_frame(session: Session, league_id: int) -> pd.DataFrame:
    builder = ChronologicalFeatureBuilder()
    records: list[dict] = []
    for match in _ordered_matches(session, league_id):
        row = builder.build_row(match.home_team_id, match.away_team_id, match.match_date)
        row.update(
            {
                "match_id": match.id,
                "match_date": match.match_date,
                "home_goals": match.home_goals,
                "away_goals": match.away_goals,
                "outcome": "H" if match.home_goals > match.away_goals else "A" if match.away_goals > match.home_goals else "D",
            }
        )
        records.append(row)
        builder.apply_result(
            match.home_team_id,
            match.away_team_id,
            match.match_date,
            match.home_goals,
            match.away_goals,
        )
    return pd.DataFrame.from_records(records, columns=FEATURE_COLUMNS + ["match_id", "match_date", "home_goals", "away_goals", "outcome"])


def build_fixture_features(
    session: Session,
    league_id: int,
    home_team_id: int,
    away_team_id: int,
    fixture_date: date,
) -> dict[str, float]:
    builder = ChronologicalFeatureBuilder()
    for match in _ordered_matches(session, league_id, before=fixture_date):
        builder.apply_result(
            match.home_team_id,
            match.away_team_id,
            match.match_date,
            match.home_goals,
            match.away_goals,
        )
    return builder.build_row(home_team_id, away_team_id, fixture_date)

