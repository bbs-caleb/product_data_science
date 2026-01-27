
import pandas as pd

from src.core.feasible_plan import compute_feasible_plan


def test_feasible_plan_floor_and_stock_cap():
    df = pd.DataFrame({
        "date": ["2025-12-01"],
        "sku": [100],
        "category": ["beverages"],
        "price": [100.0],
        "stock": [3],
        "forecast_gmv": [400.0],
    })
    out = compute_feasible_plan(df)
    assert int(out.loc[0, "forecast_units"]) == 4
    assert int(out.loc[0, "feasible_units"]) == 3
    assert float(out.loc[0, "feasible_gmv"]) == 300.0
    assert bool(out.loc[0, "is_stock_constrained"]) is True
