"""Step 02: Statistical tests on residual distributions."""

from typing import Tuple, Optional
import numpy as np
from scipy import stats


def test_normality(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Shapiro-Wilk normality test on residuals.

    Returns
    -------
    p_value : float
    is_rejected : bool
        True if normality hypothesis is rejected.
    """
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    _, p_value = stats.shapiro(residuals)
    return p_value, p_value < alpha


def test_unbiased(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefer: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """One-sample t-test checking if mean residual equals zero.

    Returns
    -------
    p_value : float
    is_rejected : bool
        True if unbiasedness hypothesis is rejected.
    """
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)

    # map prefer to scipy alternative parameter
    if prefer is None or prefer == "two-sided":
        alternative = "two-sided"
    elif prefer == "positive":
        # H0: mean=0, H1: mean>0 — reject only if residuals are negative
        alternative = "greater"
    elif prefer == "negative":
        alternative = "less"
    else:
        raise ValueError(f"Unknown prefer value: {prefer}")

    _, p_value = stats.ttest_1samp(residuals, 0, alternative=alternative)
    return p_value, p_value < alpha


def test_homoscedasticity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: int = 10,
    test: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Variance equality test across target-sorted bins of residuals.

    Returns
    -------
    p_value : float
    is_rejected : bool
        True if homoscedasticity hypothesis is rejected.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    # sort residuals by target value
    order = np.argsort(y_true)
    residuals_sorted = residuals[order]

    # split into equal-width bins, last bin gets the remainder
    n = len(residuals_sorted)
    bin_size = n // bins
    groups = []
    for i in range(bins):
        start = i * bin_size
        # last bin absorbs remainder
        end = (i + 1) * bin_size if i < bins - 1 else n
        groups.append(residuals_sorted[start:end])

    # choose test
    if test is None or test == "bartlett":
        _, p_value = stats.bartlett(*groups)
    elif test == "levene":
        _, p_value = stats.levene(*groups)
    elif test == "fligner":
        _, p_value = stats.fligner(*groups)
    else:
        raise ValueError(f"Unknown test: {test}")

    return p_value, p_value < alpha
