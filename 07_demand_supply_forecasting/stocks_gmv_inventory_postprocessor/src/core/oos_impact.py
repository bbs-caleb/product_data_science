
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_lost_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate lost sales due to stockouts (practical proxy).

    We treat the demand forecast (in units) as a proxy for unconstrained demand.
    If stock is below forecast_units, the gap is counted as 'lost'.

    Adds:
    - lost_sales_units
    - lost_sales_gmv
    """
    out = df.copy()

    forecast_units = out["forecast_units"].to_numpy(dtype=np.int64, copy=False)
    stock = out["stock"].to_numpy(dtype=np.int64, copy=False)
    price = out["price"].to_numpy(dtype=np.float64, copy=False)

    lost_units = np.maximum(forecast_units - stock, 0)
    out["lost_sales_units"] = lost_units
    out["lost_sales_gmv"] = lost_units * price
    return out
