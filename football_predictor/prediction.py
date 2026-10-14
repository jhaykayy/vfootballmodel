from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from .exceptions import ModelNotTrainedError, ValidationError
from .features import build_fixture_features
from .ml import ModelBundle, predict_one
from .models import ModelVersion, Prediction, UpcomingFixture
from .probabilities import OUTCOME_ORDER, score_summary


def get_active_model(session: Session, league_id: int) -> ModelVersion:
    version = session.scalar(
        select(ModelVersion).where(
            ModelVersion.league_id == league_id, ModelVersion.is_active.is_(True)
        )
    )
    if version is None:
        raise ModelNotTrainedError(
            "No trained model is active for this league. Add historical results and retrain first."
        )
    if not Path(version.artifact_path).exists():
        raise ModelNotTrainedError("The active model artifact is missing. Retrain this league.")
    return version


def _reliability(features: dict[str, float], bundle: ModelBundle) -> tuple[str, list[str]]:
    minimum_history = int(min(features["home_matches_played"], features["away_matches_played"]))
    notes: list[str] = []
    if minimum_history < 5:
        notes.append("At least one team has fewer than five prior matches; league defaults have substantial influence.")
        return "limited", notes
    if minimum_history < 15:
        notes.append("Team-level history is still limited, so uncertainty is elevated.")
        return "moderate", notes
    if bundle.training_matches < 100:
        notes.append("The league model has fewer than 100 training matches.")
        return "moderate", notes
    notes.append("Both teams have established histories in the fitted league dataset.")
    return "established", notes


def _explanation(features: dict[str, float], bundle: ModelBundle) -> tuple[dict, str]:
    reliability, warnings = _reliability(features, bundle)
    form_delta = features["home_points_5"] - features["away_points_5"]
    venue_delta = features["home_venue_points_10"] - features["away_venue_points_10"]
    factors = [
        f"Recent form: home {features['home_points_5']:.2f} vs away {features['away_points_5']:.2f} points per match over up to five games.",
        f"Venue form: home {features['home_venue_points_10']:.2f} vs away {features['away_venue_points_10']:.2f} points per relevant venue match.",
        f"Elo: home {features['home_elo']:.0f}, away {features['away_elo']:.0f} (difference {features['elo_diff']:+.0f}).",
        f"League scoring before kickoff: {features['league_home_goals']:.2f} home and {features['league_away_goals']:.2f} away goals per match.",
    ]
    if features["h2h_count"] >= 3:
        factors.append(
            f"Head-to-head: {int(features['h2h_count'])} recent meetings, lightly weighted alongside broader form."
        )
    else:
        factors.append("Head-to-head history is sparse and is not treated as a strong signal.")
    if abs(form_delta) > 0.5:
        factors.append("Recent form favors the home side." if form_delta > 0 else "Recent form favors the away side.")
    if abs(venue_delta) > 0.5:
        factors.append("Venue-specific results favor the home side." if venue_delta > 0 else "Venue-specific results favor the away side.")
    return {"factors": factors, "data_warnings": warnings}, reliability


def generate_prediction(session: Session, fixture_id: int) -> Prediction:
    fixture = session.get(UpcomingFixture, fixture_id)
    if fixture is None:
        raise ValidationError("Fixture not found.")
    if fixture.status == "completed":
        raise ValidationError("This fixture is already completed.")
    version = get_active_model(session, fixture.league_id)
    if fixture.match_date <= version.training_cutoff:
        raise ValidationError(
            "This fixture predates the active model's training cutoff; using it would leak future information."
        )
    bundle: ModelBundle = joblib.load(version.artifact_path)
    features = build_fixture_features(
        session,
        fixture.league_id,
        fixture.home_team_id,
        fixture.away_team_id,
        fixture.match_date,
    )
    outcome_probs, matrix = predict_one(bundle, features)
    summary = score_summary(matrix)
    summary["top_scores_json"] = summary.pop("top_scores")
    explanation, reliability = _explanation(features, bundle)
    predicted_outcome = OUTCOME_ORDER[int(np.argmax(outcome_probs))]
    prediction = Prediction(
        fixture_id=fixture.id,
        model_version_id=version.id,
        predicted_outcome=predicted_outcome,
        home_win_probability=float(outcome_probs[2]),
        draw_probability=float(outcome_probs[1]),
        away_win_probability=float(outcome_probs[0]),
        reliability=reliability,
        explanation_json=explanation,
        feature_snapshot_json={key: float(value) for key, value in features.items()},
        **summary,
    )
    session.add(prediction)
    session.flush()
    return prediction


def generate_bulk_predictions(session: Session, fixture_ids: list[int]) -> list[Prediction]:
    return [generate_prediction(session, fixture_id) for fixture_id in fixture_ids]

