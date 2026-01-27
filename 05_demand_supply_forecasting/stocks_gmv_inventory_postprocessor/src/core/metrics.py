
from __future__ import annotations

import numpy as np
import pandas as pd


def wape(actual: pd.Series, forecast: pd.Series) -> float:
    denom = float(np.sum(np.abs(actual)))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(actual - forecast)) / denom)


def bias(actual: pd.Series, forecast: pd.Series) -> float:
    denom = float(np.sum(actual))
    if denom == 0:
        return 0.0
    return float(np.sum(forecast - actual) / denom)


def compute_quality_metrics(df: pd.DataFrame) -> dict:
    """Compute quality metrics with a non-OOS split.

    If actual_units is present:
    - raw metrics: on all rows
    - non-OOS metrics: only where is_stock_constrained == False
    """
    if "actual_units" not in df.columns:
        return {}

    actual = df["actual_units"].astype(float)
    forecast = df["forecast_units"].astype(float)

    metrics = {
        "wape_raw": wape(actual, forecast),
        "bias_raw": bias(actual, forecast),
    }

    non_oos = df.loc[~df["is_stock_constrained"]]
    if len(non_oos) > 0:
        a2 = non_oos["actual_units"].astype(float)
        f2 = non_oos["forecast_units"].astype(float)
        metrics["wape_non_oos"] = wape(a2, f2)
        metrics["bias_non_oos"] = bias(a2, f2)
    else:
        metrics["wape_non_oos"] = 0.0
        metrics["bias_non_oos"] = 0.0

    return metrics
