
import pandas as pd

from src.core.feasible_plan import compute_feasible_plan
from src.core.oos_impact import compute_lost_sales


def test_lost_sales_units_and_gmv():
    df = pd.DataFrame({
        "date": ["2025-12-01"],
        "sku": [100],
        "category": ["beverages"],
        "price": [100.0],
        "stock": [3],
        "forecast_gmv": [400.0],
    })
    out = compute_lost_sales(compute_feasible_plan(df))
    assert int(out.loc[0, "lost_sales_units"]) == 1
    assert float(out.loc[0, "lost_sales_gmv"]) == 100.0
