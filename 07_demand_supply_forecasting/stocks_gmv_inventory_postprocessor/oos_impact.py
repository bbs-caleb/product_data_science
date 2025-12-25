import numpy as np
import pandas as pd


def build_feasible_sales_plan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a GMV forecast into a feasible plan under inventory constraints.

    Business logic:
    - Forecast is provided in money (GMV), but sales happen in integer units.
    - Units forecast = floor(gmv / price).
    - Feasible units cannot exceed available stock.
    - Feasible GMV = feasible_units * price.
    - Lost sales proxy quantifies the gap caused by insufficient stock.

    Expected columns:
      - gmv (float): forecast GMV
      - price (float): unit price
      - stock (int): available units
    """
    out = df.copy()

    gmv = out["gmv"].to_numpy(dtype=np.float64, copy=False)
    price = out["price"].to_numpy(dtype=np.float64, copy=False)
    stock = out["stock"].to_numpy(dtype=np.int64, copy=False)

    units_float = np.zeros_like(gmv, dtype=np.float64)
    np.divide(gmv, price, out=units_float, where=price > 0)

    units_forecast = np.floor(units_float).astype(np.int64, copy=False)
    units_forecast = np.maximum(units_forecast, 0)

    units_feasible = np.minimum(units_forecast, stock)

    out["units_forecast"] = units_forecast
    out["units_feasible"] = units_feasible
    out["gmv_feasible"] = units_feasible * price

    lost_units = units_forecast - units_feasible
    out["lost_sales_units"] = np.maximum(lost_units, 0)
    out["lost_sales_gmv"] = out["lost_sales_units"] * out["price"]

    return out
