import numpy as np


def smape(y_true: np.array, y_pred: np.array) -> float:
    """smape function implementation"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    diff = np.abs(y_true - y_pred) * 2
    denom = np.abs(y_true) + np.abs(y_pred)

    out = np.zeros_like(denom, dtype=float)

    np.divide(diff, denom, out=out, where=denom != 0)

    return float(out.mean())
