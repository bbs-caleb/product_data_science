
import pandas as pd

from src.core.metrics import compute_quality_metrics


def test_metrics_non_oos_split():
    df = pd.DataFrame({
        "actual_units": [10, 10],
        "forecast_units": [12, 8],
        "is_stock_constrained": [False, True],
    })
    m = compute_quality_metrics(df)
    assert "wape_raw" in m and "wape_non_oos" in m
    # non-OOS subset includes only first row: abs(10-12)/10 = 0.2
    assert abs(m["wape_non_oos"] - 0.2) < 1e-9
