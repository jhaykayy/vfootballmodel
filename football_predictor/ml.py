from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS
from .probabilities import (
    OUTCOME_ORDER,
    align_score_matrix_to_outcomes,
    outcome_probabilities_from_matrix,
    poisson_score_matrix,
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - dependency is declared, fallback keeps diagnostics usable
    XGBClassifier = None  # type: ignore[assignment]


@dataclass
class ModelBundle:
    feature_columns: list[str]
    home_goal_model: Pipeline
    away_goal_model: Pipeline
    outcome_model: Optional[Any]
    outcome_model_name: str
    outcome_weight: float
    max_score: int
    training_matches: int
    metrics: dict[str, float]


def _goal_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("poisson", PoissonRegressor(alpha=0.35, max_iter=1000)),
        ]
    )


def _outcome_estimator(random_state: int):
    if XGBClassifier is not None:
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=160,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            min_child_weight=3,
            random_state=random_state,
            n_jobs=2,
            eval_metric="mlogloss",
        ), "xgboost_calibrated"
    return HistGradientBoostingClassifier(
        max_iter=160, max_depth=3, learning_rate=0.04, l2_regularization=2.0, random_state=random_state
    ), "sklearn_hist_gradient_boosting_calibrated"


def _encode_outcomes(values: pd.Series) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(OUTCOME_ORDER)}
    return values.map(mapping).to_numpy(dtype=int)


def _can_fit_outcome(y: np.ndarray, minimum: int = 36) -> bool:
    if len(y) < minimum:
        return False
    counts = np.bincount(y, minlength=3)
    return bool(np.all(counts >= 4))


def _fit_calibrated_outcome(
    X: pd.DataFrame, y: np.ndarray, random_state: int
) -> tuple[Optional[Any], str]:
    if not _can_fit_outcome(y):
        return None, "poisson_only"
    calibration_size = max(9, int(round(len(X) * 0.18)))
    split = len(X) - calibration_size
    if split < 24 or len(np.unique(y[:split])) < 3 or len(np.unique(y[split:])) < 3:
        return None, "poisson_only"
    estimator, name = _outcome_estimator(random_state)
    estimator.fit(X.iloc[:split], y[:split])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated = CalibratedClassifierCV(estimator, method="sigmoid", cv="prefit")
        calibrated.fit(X.iloc[split:], y[split:])
    return calibrated, name


def elo_outcome_probabilities(feature_frame: pd.DataFrame) -> np.ndarray:
    diff = feature_frame["elo_diff"].to_numpy(dtype=float) + 60.0
    home_share = 1.0 / (1.0 + np.power(10.0, -diff / 400.0))
    draw = np.clip(0.285 - np.abs(diff) / 1800.0, 0.17, 0.285)
    home = (1.0 - draw) * home_share
    away = (1.0 - draw) * (1.0 - home_share)
    return np.column_stack([away, draw, home])


def _model_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    result = np.full((len(X), 3), 1e-9, dtype=float)
    for source_index, class_index in enumerate(classes):
        result[:, int(class_index)] = raw[:, source_index]
    return result / result.sum(axis=1, keepdims=True)


def _poisson_probabilities(home_model: Pipeline, away_model: Pipeline, X: pd.DataFrame, max_score: int) -> np.ndarray:
    home_expected = np.clip(home_model.predict(X), 0.05, 6.0)
    away_expected = np.clip(away_model.predict(X), 0.05, 6.0)
    return np.vstack(
        [
            outcome_probabilities_from_matrix(poisson_score_matrix(home, away, max_score))
            for home, away in zip(home_expected, away_expected)
        ]
    )


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return {
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(y_true, probabilities.argmax(axis=1))),
        "brier_score": float(
            np.mean(
                [
                    brier_score_loss((y_true == class_index).astype(int), probabilities[:, class_index])
                    for class_index in range(3)
                ]
            )
        ),
    }


def fit_bundle(
    frame: pd.DataFrame,
    *,
    train_end: int,
    validation_end: int,
    max_score: int,
    random_state: int,
) -> tuple[ModelBundle, dict[str, dict[str, float]]]:
    if not (10 <= train_end < validation_end <= len(frame)):
        raise ValueError("Invalid chronological split boundaries.")
    features = frame[FEATURE_COLUMNS].astype(float)
    outcomes = _encode_outcomes(frame["outcome"])
    train_X = features.iloc[:train_end]
    val_X = features.iloc[train_end:validation_end]
    fit_X = features.iloc[:validation_end]
    test_X = features.iloc[validation_end:]
    train_y, val_y = outcomes[:train_end], outcomes[train_end:validation_end]
    fit_y, test_y = outcomes[:validation_end], outcomes[validation_end:]

    validation_home = _goal_model().fit(train_X, frame["home_goals"].iloc[:train_end])
    validation_away = _goal_model().fit(train_X, frame["away_goals"].iloc[:train_end])
    poisson_val = _poisson_probabilities(validation_home, validation_away, val_X, max_score)
    elo_val = elo_outcome_probabilities(val_X)
    validation: dict[str, dict[str, float]] = {
        "poisson": classification_metrics(val_y, poisson_val),
        "elo": classification_metrics(val_y, elo_val),
    }
    train_frequency = (np.bincount(train_y, minlength=3).astype(float) + 1.0) / (len(train_y) + 3.0)
    validation["historical_frequency"] = classification_metrics(
        val_y, np.tile(train_frequency, (len(val_y), 1))
    )

    outcome_model, outcome_name = _fit_calibrated_outcome(train_X, train_y, random_state)
    best_weight = 0.0
    if outcome_model is not None:
        outcome_val = _model_probabilities(outcome_model, val_X)
        validation[outcome_name] = classification_metrics(val_y, outcome_val)
        candidates = []
        for weight in (0.25, 0.5, 0.75, 1.0):
            combined = weight * outcome_val + (1.0 - weight) * poisson_val
            candidates.append((classification_metrics(val_y, combined)["log_loss"], weight))
        best_weight = min(candidates)[1]
        validation["ensemble"] = classification_metrics(
            val_y, best_weight * outcome_val + (1.0 - best_weight) * poisson_val
        )
    else:
        validation["ensemble"] = dict(validation["poisson"])

    home_model = _goal_model().fit(fit_X, frame["home_goals"].iloc[:validation_end])
    away_model = _goal_model().fit(fit_X, frame["away_goals"].iloc[:validation_end])
    final_outcome_model, final_outcome_name = _fit_calibrated_outcome(fit_X, fit_y, random_state)
    if final_outcome_model is None:
        best_weight = 0.0
        final_outcome_name = "poisson_only"
    bundle = ModelBundle(
        feature_columns=list(FEATURE_COLUMNS),
        home_goal_model=home_model,
        away_goal_model=away_model,
        outcome_model=final_outcome_model,
        outcome_model_name=final_outcome_name,
        outcome_weight=best_weight,
        max_score=max_score,
        training_matches=validation_end,
        metrics={},
    )
    if len(test_X):
        test_models = predict_probability_components(bundle, test_X)
        for name, probabilities in test_models.items():
            validation[f"test_{name}"] = classification_metrics(test_y, probabilities)
        fit_frequency = (np.bincount(fit_y, minlength=3).astype(float) + 1.0) / (len(fit_y) + 3.0)
        validation["test_historical_frequency"] = classification_metrics(
            test_y, np.tile(fit_frequency, (len(test_y), 1))
        )
        bundle.metrics = validation["test_ensemble"]
    else:
        bundle.metrics = validation["poisson"]
    return bundle, validation


def predict_probability_components(bundle: ModelBundle, X: pd.DataFrame) -> dict[str, np.ndarray]:
    poisson_probs = _poisson_probabilities(bundle.home_goal_model, bundle.away_goal_model, X, bundle.max_score)
    result = {
        "poisson": poisson_probs,
        "elo": elo_outcome_probabilities(X),
    }
    if bundle.outcome_model is not None and bundle.outcome_weight > 0:
        outcome_probs = _model_probabilities(bundle.outcome_model, X)
        result[bundle.outcome_model_name] = outcome_probs
        result["ensemble"] = (
            bundle.outcome_weight * outcome_probs + (1.0 - bundle.outcome_weight) * poisson_probs
        )
    else:
        result["ensemble"] = poisson_probs
    return result


def predict_one(bundle: ModelBundle, features: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    X = pd.DataFrame([{name: features[name] for name in bundle.feature_columns}])
    home_expected = float(np.clip(bundle.home_goal_model.predict(X)[0], 0.05, 6.0))
    away_expected = float(np.clip(bundle.away_goal_model.predict(X)[0], 0.05, 6.0))
    raw_matrix = poisson_score_matrix(home_expected, away_expected, bundle.max_score)
    final_probs = predict_probability_components(bundle, X)["ensemble"][0]
    return final_probs / final_probs.sum(), align_score_matrix_to_outcomes(raw_matrix, final_probs)


def refit_bundle_for_deployment(
    evaluated_bundle: ModelBundle, frame: pd.DataFrame, *, random_state: int
) -> ModelBundle:
    """Refit after evaluation so the deployed artifact uses every completed match."""
    X = frame[FEATURE_COLUMNS].astype(float)
    y = _encode_outcomes(frame["outcome"])
    home_model = _goal_model().fit(X, frame["home_goals"])
    away_model = _goal_model().fit(X, frame["away_goals"])
    outcome_model, outcome_name = _fit_calibrated_outcome(X, y, random_state)
    weight = evaluated_bundle.outcome_weight if outcome_model is not None else 0.0
    return ModelBundle(
        feature_columns=list(FEATURE_COLUMNS),
        home_goal_model=home_model,
        away_goal_model=away_model,
        outcome_model=outcome_model,
        outcome_model_name=outcome_name,
        outcome_weight=weight,
        max_score=evaluated_bundle.max_score,
        training_matches=len(frame),
        metrics=dict(evaluated_bundle.metrics),
    )

