import numpy as np

from football_predictor.probabilities import (
    align_score_matrix_to_outcomes,
    outcome_probabilities_from_matrix,
    poisson_score_matrix,
    score_summary,
)


def test_probability_matrix_is_internally_consistent():
    matrix = poisson_score_matrix(1.7, 1.1, max_score=10)
    target = np.array([0.20, 0.25, 0.55])
    adjusted = align_score_matrix_to_outcomes(matrix, target)
    actual = outcome_probabilities_from_matrix(adjusted)
    assert adjusted.sum() == pytest.approx(1.0)
    assert actual == pytest.approx(target)
    summary = score_summary(adjusted)
    assert summary["over_25_probability"] + summary["under_25_probability"] == pytest.approx(1.0)
    assert summary["btts_yes_probability"] + summary["btts_no_probability"] == pytest.approx(1.0)
    assert len(summary["top_scores"]) == 3


import pytest

