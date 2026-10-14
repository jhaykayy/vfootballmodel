from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .exceptions import FootballPredictorError, InsufficientDataError, ModelTrainingError
from .features import FEATURE_COLUMNS, generate_training_frame
from .ml import (
    ModelBundle,
    classification_metrics,
    fit_bundle,
    predict_probability_components,
    refit_bundle_for_deployment,
)
from .model_storage import load_model_bundle, model_bundle_exists, save_model_bundle
from .models import League, ModelMetric, ModelVersion, TrainingRun


def _splits(size: int) -> tuple[int, int]:
    train_end = max(10, int(size * 0.70))
    validation_end = max(train_end + 3, int(size * 0.85))
    validation_end = min(validation_end, size - 3)
    if validation_end <= train_end:
        train_end = max(10, size - 6)
        validation_end = size - 3
    return train_end, validation_end


def walk_forward_backtest(frame, settings: Settings) -> dict[str, float]:
    """Expanding-window backtest; each block is predicted by a model fit only on prior rows."""
    if len(frame) < 45:
        return {}
    minimum_train = max(30, int(len(frame) * 0.45))
    remaining = len(frame) - minimum_train
    block_size = max(5, remaining // 4)
    truth: list[int] = []
    probabilities: list[np.ndarray] = []
    label_map = {"A": 0, "D": 1, "H": 2}
    for start in range(minimum_train, len(frame), block_size):
        history = frame.iloc[:start].reset_index(drop=True)
        end = min(start + block_size, len(frame))
        future = frame.iloc[start:end]
        train_end, validation_end = _splits(len(history))
        evaluated, _ = fit_bundle(
            history,
            train_end=train_end,
            validation_end=validation_end,
            max_score=settings.max_score,
            random_state=settings.random_state,
        )
        fitted = refit_bundle_for_deployment(evaluated, history, random_state=settings.random_state)
        fold_probs = predict_probability_components(fitted, future[FEATURE_COLUMNS].astype(float))["ensemble"]
        probabilities.extend(fold_probs)
        truth.extend(label_map[value] for value in future["outcome"])
    return classification_metrics(np.asarray(truth), np.asarray(probabilities))


def _evaluate_existing(bundle: ModelBundle, test_frame) -> Optional[dict[str, float]]:
    try:
        labels = test_frame["outcome"].map({"A": 0, "D": 1, "H": 2}).to_numpy(dtype=int)
        probs = predict_probability_components(bundle, test_frame[FEATURE_COLUMNS].astype(float))["ensemble"]
        return classification_metrics(labels, probs)
    except Exception:
        return None


def train_league(
    session: Session,
    league_id: int,
    *,
    trigger: str = "manual",
    settings: Optional[Settings] = None,
) -> ModelVersion:
    settings = settings or get_settings()
    league = session.get(League, league_id)
    if league is None:
        raise ValueError("League not found.")
    run = TrainingRun(league_id=league_id, trigger=trigger, status="running")
    session.add(run)
    session.flush()
    try:
        frame = generate_training_frame(session, league_id)
        if len(frame) < settings.min_training_matches:
            raise InsufficientDataError(
                f"{league.name} needs at least {settings.min_training_matches} completed matches; "
                f"it currently has {len(frame)}."
            )
        train_end, validation_end = _splits(len(frame))
        evaluated_bundle, evaluations = fit_bundle(
            frame,
            train_end=train_end,
            validation_end=validation_end,
            max_score=settings.max_score,
            random_state=settings.random_state,
        )
        test_frame = frame.iloc[validation_end:]
        candidate_test = evaluations["test_ensemble"]
        current = session.scalar(
            select(ModelVersion).where(
                ModelVersion.league_id == league_id, ModelVersion.is_active.is_(True)
            )
        )
        previous_on_current_test = None
        if current and model_bundle_exists(session, current):
            previous_bundle = load_model_bundle(session, current)
            previous_on_current_test = _evaluate_existing(previous_bundle, test_frame)
        promote = current is None
        if previous_on_current_test is not None:
            # Tiny changes are noise; require a real or practically equivalent result.
            promote = candidate_test["log_loss"] <= previous_on_current_test["log_loss"] + 0.002
        elif current is not None:
            promote = candidate_test["log_loss"] <= current.test_log_loss + 0.002

        backtest = walk_forward_backtest(frame, settings)
        deployed_bundle = refit_bundle_for_deployment(
            evaluated_bundle, frame, random_state=settings.random_state
        )
        version_name = f"{league.code}-{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"
        model_version = ModelVersion(
            league_id=league_id,
            version=version_name,
            algorithm=(
                f"Poisson + {deployed_bundle.outcome_model_name}"
                if deployed_bundle.outcome_weight
                else "Poisson regression"
            ),
            artifact_path="pending",
            training_cutoff=frame["match_date"].max(),
            training_matches=len(frame),
            validation_log_loss=float(evaluations["ensemble"]["log_loss"]),
            test_log_loss=float(candidate_test["log_loss"]),
            is_active=promote,
            promoted=promote,
            metadata_json={
                "features": list(FEATURE_COLUMNS),
                "outcome_weight": deployed_bundle.outcome_weight,
                "validation": evaluations,
                "walk_forward": backtest,
                "previous_model_on_test": previous_on_current_test,
                "split": {
                    "train_rows": train_end,
                    "validation_rows": validation_end - train_end,
                    "test_rows": len(frame) - validation_end,
                },
            },
        )
        session.add(model_version)
        session.flush()
        artifact_path = settings.model_dir / f"{version_name}.joblib"
        save_model_bundle(session, model_version, deployed_bundle, artifact_path)
        model_version.artifact_path = str(artifact_path)
        if promote:
            session.execute(
                update(ModelVersion)
                .where(ModelVersion.league_id == league_id, ModelVersion.id != model_version.id)
                .values(is_active=False)
            )
        for model_name, metric_values in evaluations.items():
            split_name = "test" if model_name.startswith("test_") else "validation"
            clean_name = model_name.removeprefix("test_")
            for metric_name, metric_value in metric_values.items():
                session.add(
                    ModelMetric(
                        model_version_id=model_version.id,
                        split=split_name,
                        model_name=clean_name,
                        metric_name=metric_name,
                        metric_value=float(metric_value),
                    )
                )
        for metric_name, metric_value in backtest.items():
            session.add(
                ModelMetric(
                    model_version_id=model_version.id,
                    split="walk_forward",
                    model_name="ensemble",
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                )
            )
        run.model_version_id = model_version.id
        run.status = "completed"
        run.finished_at = datetime.utcnow()
        run.promoted = promote
        run.message = "Candidate promoted." if promote else "Candidate retained for audit; active model performed better."
        session.flush()
        return model_version
    except Exception as exc:
        run.status = "failed"
        run.finished_at = datetime.utcnow()
        run.message = str(exc)
        session.flush()
        if isinstance(exc, FootballPredictorError):
            raise
        raise ModelTrainingError(f"Model training failed: {exc}") from exc


def train_all_eligible(session: Session, *, trigger: str = "manual") -> list[ModelVersion]:
    versions: list[ModelVersion] = []
    for league_id in session.scalars(select(League.id).order_by(League.id)).all():
        try:
            versions.append(train_league(session, league_id, trigger=trigger))
        except InsufficientDataError:
            continue
    return versions

