
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_feasible_plan(df: pd.DataFrame) -> pd.DataFrame:
    """Create an executable (supply-constrained) plan from a GMV forecast.

    Business logic (retail planning):
    - Forecast is given in GMV (money), but execution is in integer units.
    - Convert GMV -> units via floor(gmv / price) to avoid over-promising.
    - Cap units by available stock (ATS / on-hand proxy).
    - Recompute feasible GMV using price * feasible_units.

    Returns a new DataFrame with:
    - forecast_units
    - feasible_units
    - feasible_gmv
    - is_stock_constrained
    """
    out = df.copy()

    gmv = out["forecast_gmv"].to_numpy(dtype=np.float64, copy=False)
    price = out["price"].to_numpy(dtype=np.float64, copy=False)
    stock = out["stock"].to_numpy(dtype=np.int64, copy=False)

    units_float = np.zeros_like(gmv, dtype=np.float64)
    np.divide(gmv, price, out=units_float, where=price > 0)

    forecast_units = np.floor(units_float).astype(np.int64, copy=False)
    forecast_units = np.maximum(forecast_units, 0)

    feasible_units = np.minimum(forecast_units, stock)
    feasible_units = np.maximum(feasible_units, 0)

    out["forecast_units"] = forecast_units
    out["feasible_units"] = feasible_units
    out["feasible_gmv"] = feasible_units * price
    out["is_stock_constrained"] = forecast_units > stock

    return out
