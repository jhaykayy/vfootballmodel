from __future__ import annotations

from collections import defaultdict

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import (
    CompletedMatch,
    League,
    ModelVersion,
    Prediction,
    PredictionOutcome,
    UpcomingFixture,
)


def dashboard_summary(session: Session) -> dict:
    historical_matches = session.scalar(select(func.count(CompletedMatch.id))) or 0
    predictions = session.scalar(select(func.count(Prediction.id))) or 0
    evaluated = session.scalar(select(func.count(PredictionOutcome.id))) or 0
    correct = session.scalar(
        select(func.count(PredictionOutcome.id)).where(PredictionOutcome.outcome_correct.is_(True))
    ) or 0
    latest_training = session.scalar(select(func.max(ModelVersion.created_at)))
    return {
        "historical_matches": int(historical_matches),
        "predictions": int(predictions),
        "evaluated_predictions": int(evaluated),
        "accuracy": float(correct / evaluated) if evaluated else None,
        "latest_training": latest_training,
    }


def performance_rows(session: Session) -> list[dict]:
    query = (
        select(PredictionOutcome, Prediction, UpcomingFixture, League)
        .join(Prediction, PredictionOutcome.prediction_id == Prediction.id)
        .join(UpcomingFixture, Prediction.fixture_id == UpcomingFixture.id)
        .join(League, UpcomingFixture.league_id == League.id)
        .order_by(PredictionOutcome.evaluated_at)
    )
    return [
        {
            "date": fixture.match_date,
            "league": league.name,
            "predicted": prediction.predicted_outcome,
            "actual": outcome.actual_outcome,
            "home_probability": prediction.home_win_probability,
            "draw_probability": prediction.draw_probability,
            "away_probability": prediction.away_win_probability,
            "outcome_correct": outcome.outcome_correct,
            "exact_score_correct": outcome.exact_score_correct,
            "btts_correct": outcome.btts_correct,
            "over_25_correct": outcome.over_25_correct,
            "brier_score": outcome.brier_score,
            "log_loss": outcome.log_loss,
        }
        for outcome, prediction, fixture, league in session.execute(query).all()
    ]


def accuracy_by_league(session: Session) -> list[dict]:
    grouped: dict[str, list] = defaultdict(list)
    for row in performance_rows(session):
        grouped[row["league"]].append(row)
    result = []
    for league, rows in grouped.items():
        result.append(
            {
                "league": league,
                "predictions": len(rows),
                "1x2_accuracy": sum(item["outcome_correct"] for item in rows) / len(rows),
                "exact_score_accuracy": sum(item["exact_score_correct"] for item in rows) / len(rows),
                "btts_accuracy": sum(item["btts_correct"] for item in rows) / len(rows),
                "over_25_accuracy": sum(item["over_25_correct"] for item in rows) / len(rows),
                "brier_score": sum(item["brier_score"] for item in rows) / len(rows),
                "log_loss": sum(item["log_loss"] for item in rows) / len(rows),
            }
        )
    return result

