from __future__ import annotations

import numpy as np
from scipy.stats import poisson


OUTCOME_ORDER = ("A", "D", "H")


def poisson_score_matrix(home_expected: float, away_expected: float, max_score: int = 10) -> np.ndarray:
    home_expected = float(np.clip(home_expected, 0.05, 6.0))
    away_expected = float(np.clip(away_expected, 0.05, 6.0))
    scores = np.arange(max_score + 1)
    matrix = np.outer(poisson.pmf(scores, home_expected), poisson.pmf(scores, away_expected))
    total = matrix.sum()
    if total <= 0:
        raise ValueError("Poisson score matrix has no probability mass.")
    return matrix / total


def outcome_probabilities_from_matrix(matrix: np.ndarray) -> np.ndarray:
    # np.triu is away goals > home goals; np.tril is home goals > away goals.
    away = float(np.triu(matrix, 1).sum())
    draw = float(np.trace(matrix))
    home = float(np.tril(matrix, -1).sum())
    values = np.asarray([away, draw, home], dtype=float)
    return values / values.sum()


def align_score_matrix_to_outcomes(matrix: np.ndarray, outcome_probs: np.ndarray) -> np.ndarray:
    """Scale A/D/H regions so score and 1X2 probabilities share exact marginals."""
    target = np.asarray(outcome_probs, dtype=float)
    target = np.clip(target, 1e-9, None)
    target /= target.sum()
    adjusted = matrix.copy()
    masks = [
        np.triu(np.ones_like(matrix, dtype=bool), 1),
        np.eye(matrix.shape[0], dtype=bool),
        np.tril(np.ones_like(matrix, dtype=bool), -1),
    ]
    for probability, mask in zip(target, masks):
        current = float(adjusted[mask].sum())
        if current > 0:
            adjusted[mask] *= probability / current
    return adjusted / adjusted.sum()


def score_summary(matrix: np.ndarray) -> dict:
    ranked = np.argsort(matrix.ravel())[::-1][:3]
    top_scores = []
    width = matrix.shape[1]
    for flat_index in ranked:
        home_goals, away_goals = divmod(int(flat_index), width)
        top_scores.append(
            {
                "score": f"{home_goals}-{away_goals}",
                "probability": float(matrix[home_goals, away_goals]),
            }
        )
    home_grid, away_grid = np.indices(matrix.shape)
    total_grid = home_grid + away_grid
    over_15 = float(matrix[total_grid >= 2].sum())
    over_25 = float(matrix[total_grid >= 3].sum())
    over_35 = float(matrix[total_grid >= 4].sum())
    btts_yes = float(matrix[(home_grid >= 1) & (away_grid >= 1)].sum())
    return {
        "expected_home_goals": float((matrix * home_grid).sum()),
        "expected_away_goals": float((matrix * away_grid).sum()),
        "most_likely_score": top_scores[0]["score"],
        "top_scores": top_scores,
        "over_15_probability": over_15,
        "over_25_probability": over_25,
        "under_25_probability": 1.0 - over_25,
        "over_35_probability": over_35,
        "btts_yes_probability": btts_yes,
        "btts_no_probability": 1.0 - btts_yes,
    }

