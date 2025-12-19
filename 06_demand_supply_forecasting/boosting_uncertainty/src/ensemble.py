"""Solution for boosting uncertainty problem."""
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class PredictionDict:
    """
    Dataclass for storing prediction results and uncertainty.
    """
    pred: np.ndarray = np.array([])
    uncertainty: np.ndarray = np.array([])
    pred_virt: np.ndarray = np.array([])
    lcb: np.ndarray = np.array([])
    ucb: np.ndarray = np.array([])


def virtual_ensemble_iterations(
    model: GradientBoostingRegressor, k: int = 20
) -> List[int]:
    """
    Return the 0-based indices of trees to include in the virtual ensemble.
    Example N=100, k=20 -> [49, 69, 89].
    Formula: range(n_estimators // 2 - 1, n_estimators, k)
    
    Note: 100 // 2 - 1 = 49. Index 49 corresponds to 50 trees.
    """
    return list(range(model.n_estimators // 2 - 1, model.n_estimators, k))


def virtual_ensemble_predict(
    model: GradientBoostingRegressor, X: np.ndarray, k: int = 20
) -> np.ndarray:
    """
    Return predictions for each model in the virtual ensemble.
    Output shape: (n_samples, n_virtual_models)
    """
    # iterations contains the 0-based indices of the stages we want
    iterations = virtual_ensemble_iterations(model, k)
    needed = set(iterations)

    preds = {}
    # staged_predict yields prediction at index 0, 1, 2...
    # We use default enumerate (starts from 0) to match the indices in 'iterations'.
    # If iterations has 49, we take the element at index 49 (which is sum of 50 trees).
    for i, pred in enumerate(model.staged_predict(X)):
        if i in needed:
            preds[i] = pred

    # Return predictions in the correct order
    result = [preds[cnt] for cnt in iterations]
    return np.column_stack(result)


def predict_with_uncertainty(
    model: GradientBoostingRegressor, X: np.ndarray, k: int = 20
) -> PredictionDict:
    """
    Return PredictionDict with virtual ensemble statistics.
    """
    stage_preds = virtual_ensemble_predict(model, X, k)

    # Mean of predictions
    pred_virt = np.mean(stage_preds, axis=1)

    # Uncertainty: Sample Standard Deviation (ddof=1).
    # This provides an unbiased estimator of the variance/std.
    uncertainty = np.std(stage_preds, axis=1, ddof=1)
    
    # Handle single model case (std is NaN for 1 element with ddof=1)
    uncertainty = np.nan_to_num(uncertainty, nan=0.0)

    # Bounds (3-sigma rule)
    lcb = pred_virt - 3 * uncertainty
    ucb = pred_virt + 3 * uncertainty

    return PredictionDict(
        pred=stage_preds,
        uncertainty=uncertainty,
        pred_virt=pred_virt,
        lcb=lcb,
        ucb=ucb
    )
