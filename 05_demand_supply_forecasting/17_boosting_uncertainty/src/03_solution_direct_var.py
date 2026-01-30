"""Solution: direct indexing staged[t], uncertainty = variance."""

from dataclasses import dataclass, field
from typing import List

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class PredictionDict:
    """Prediction results with uncertainty estimates."""

    pred: np.ndarray = field(default_factory=lambda: np.array([]))
    uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))
    pred_virt: np.ndarray = field(default_factory=lambda: np.array([]))
    lcb: np.ndarray = field(default_factory=lambda: np.array([]))
    ucb: np.ndarray = field(default_factory=lambda: np.array([]))


def virtual_ensemble_iterations(
    model: GradientBoostingRegressor, k: int = 20
) -> List[int]:
    """Return list of iteration counts for virtual ensemble."""
    n_estimators = model.n_estimators
    start = n_estimators // 2 - 1
    return list(range(start, n_estimators, k))


def virtual_ensemble_predict(
    model: GradientBoostingRegressor, X: np.ndarray, k: int = 20
) -> np.ndarray:
    """Return matrix (N, K) - DIRECT indexing staged[t]."""
    iterations = virtual_ensemble_iterations(model, k)
    if not iterations:
        return np.zeros((np.asarray(X).shape[0], 0))
    staged = list(model.staged_predict(X))
    # DIRECT indexing: staged[t], not staged[t-1]
    return np.column_stack([staged[t] for t in iterations])


def predict_with_uncertainty(
    model: GradientBoostingRegressor, X: np.ndarray, k: int = 20
) -> PredictionDict:
    """Direct indexing, uncertainty = variance."""
    pred = virtual_ensemble_predict(model, X, k)
    n_samples = pred.shape[0]
    n_models = pred.shape[1] if pred.ndim > 1 else 0

    if n_models == 0:
        zeros = np.zeros(n_samples)
        return PredictionDict(
            pred=pred,
            uncertainty=zeros,
            pred_virt=zeros,
            lcb=zeros,
            ucb=zeros,
        )

    pred_virt = np.mean(pred, axis=1)
    uncertainty = np.var(pred, axis=1)
    sqrt_unc = np.sqrt(uncertainty)
    lcb = pred_virt - 3.0 * sqrt_unc
    ucb = pred_virt + 3.0 * sqrt_unc

    return PredictionDict(
        pred=pred,
        uncertainty=uncertainty,
        pred_virt=pred_virt,
        lcb=lcb,
        ucb=ucb,
    )
