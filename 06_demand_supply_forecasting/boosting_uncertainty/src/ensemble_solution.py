"""Step 5: Virtual ensemble for GradientBoostingRegressor.

We build a *virtual ensemble* by taking predictions of the SAME trained boosting
model after a set of intermediate boosting iterations (i.e. after the first
t trees were added). This can be computed via `model.staged_predict(X)` which
correctly accounts for both `learning_rate` and the initial model `init(X)`.

Outputs:
- pred: (N, K) matrix with predictions of K virtual models
- pred_virt: (N,) mean prediction across K virtual models
- uncertainty: (N,) per-object dispersion of the K predictions (variance)
- lcb/ucb: (N,) lower/upper confidence bounds:
           pred_virt ± 3 * sqrt(uncertainty)

Notes on iteration indexing:
- A value t means "use the first t trees", i.e. tree indices [0, 1, ..., t-1].
- In `staged_predict`, the first yielded prediction corresponds to t=1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


@dataclass(frozen=True)
class PredictionDict:
    pred: np.ndarray         # shape (N, K)
    pred_virt: np.ndarray    # shape (N,)
    uncertainty: np.ndarray  # shape (N,)
    lcb: np.ndarray          # shape (N,)
    ucb: np.ndarray          # shape (N,)


def virtual_ensemble_iterations(n_estimators: int, k: int) -> List[int]:
    """Return a list of virtual-ensemble iteration sizes (tree counts).

    The course definition starts from the middle of boosting and then takes
    every k-th iteration.

    Example:
        n_estimators=100, k=20 -> [49, 69, 89]

    Edge cases:
        - Always return at least one positive integer in [1, n_estimators].
    """
    if not isinstance(n_estimators, int) or n_estimators <= 0:
        raise ValueError("n_estimators must be a positive int")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive int")

    start = max(1, n_estimators // 2 - 1)

    iters = list(range(start, n_estimators, k))
    # If the range is empty (possible for very small n_estimators), fall back.
    if not iters:
        iters = [n_estimators]

    return iters


def virtual_ensemble_predict(
    model: GradientBoostingRegressor,
    X: np.ndarray,
    iterations: Sequence[int],
) -> np.ndarray:
    """Compute prediction matrix of the virtual ensemble.

    Args:
        model: fitted GradientBoostingRegressor
        X: features, shape (N, d)
        iterations: sequence of tree counts t (see module docstring)

    Returns:
        pred: ndarray shape (N, K) where K=len(iterations)
    """
    if iterations is None or len(iterations) == 0:
        raise ValueError("iterations must be a non-empty sequence")

    # Keep order, drop duplicates
    seen = set()
    iters: List[int] = []
    for t in iterations:
        t_int = int(t)
        if t_int not in seen:
            seen.add(t_int)
            iters.append(t_int)

    n_estimators = int(getattr(model, "n_estimators", 0))
    if n_estimators <= 0:
        raise ValueError("model.n_estimators must be positive")

    # Validate bounds: t in [1, n_estimators]
    for t in iters:
        if t < 1 or t > n_estimators:
            raise ValueError(f"iteration t={t} must be in [1, {n_estimators}]")

    # Preallocate output
    n_samples = X.shape[0]
    K = len(iters)
    pred = np.empty((n_samples, K), dtype=float)

    # Map tree-count -> column index
    col_by_t = {t: j for j, t in enumerate(iters)}
    need = set(iters)
    max_t = max(iters)

    # staged_predict yields predictions after each stage:
    # stage_idx=1 corresponds to using first 1 tree, etc.
    for stage_idx, y_stage in enumerate(model.staged_predict(X), start=1):
        if stage_idx > max_t:
            break
        if stage_idx in need:
            pred[:, col_by_t[stage_idx]] = np.asarray(y_stage, dtype=float).reshape(-1)
            need.remove(stage_idx)
            if not need:
                break

    if need:
        # This should not happen, but keep it explicit for debuggability.
        missing = sorted(need)
        raise RuntimeError(f"Could not compute predictions for iterations: {missing}")

    return pred


def predict_with_uncertainty(
    model: GradientBoostingRegressor,
    X: np.ndarray,
    k: int = 20,
) -> PredictionDict:
    """Compute virtual-ensemble predictions and uncertainty estimates.

    Returns:
        PredictionDict with:
        - pred: (N, K)
        - pred_virt: (N,)
        - uncertainty: (N,) variance across K predictions
        - lcb/ucb: (N,) pred_virt ± 3*sqrt(uncertainty)
    """
    iterations = virtual_ensemble_iterations(model.n_estimators, k)
    pred = virtual_ensemble_predict(model, X, iterations)

    pred_virt = pred.mean(axis=1)

    # Uncertainty proxy: variance of virtual ensemble predictions per object
    uncertainty = pred.var(axis=1)

    sigma = np.sqrt(uncertainty)
    lcb = pred_virt - 3.0 * sigma
    ucb = pred_virt + 3.0 * sigma

    return PredictionDict(
        pred=pred,
        pred_virt=pred_virt,
        uncertainty=uncertainty,
        lcb=lcb,
        ucb=ucb,
    )
