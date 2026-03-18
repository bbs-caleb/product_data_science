"""Step 01: Residuals and pseudo-residuals for regression and classification."""

import numpy as np


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Signed residuals (MAE terms before abs)."""
    return y_true - y_pred


def squared_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MSE terms: squared difference per object."""
    return (y_true - y_pred) ** 2


def logloss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """LogLoss pseudo-residuals: per-object contribution to logloss."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # predictions must be strictly inside (0, 1)
    if np.any(y_pred <= 0) or np.any(y_pred >= 1):
        raise ValueError("y_pred must be strictly between 0 and 1")
    return y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)


def ape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MAPE pseudo-residuals: signed relative error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # mape is defined for non-negative values only
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("y_true and y_pred must be non-negative")
    return 1 - y_pred / y_true


def quantile_loss(
    y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.01
) -> np.ndarray:
    """Quantile loss terms: asymmetric penalty per object."""
    diff = y_true - y_pred
    return np.where(diff >= 0, q * diff, (1 - q) * (-diff))
