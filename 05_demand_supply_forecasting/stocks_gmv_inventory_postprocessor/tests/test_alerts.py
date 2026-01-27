
import pandas as pd

from src.alerts.rules import generate_alerts


def test_supply_alert_fires_on_losses():
    df = pd.DataFrame({
        "lost_sales_gmv": [600, 600],
        "is_stock_constrained": [True, True],
        "price": [100, 100],
    })
    metrics = {"wape_raw": 0.1, "wape_non_oos": 0.1}
    thresholds = {
        "supply": {"lost_sales_gmv_week_min": 1000, "oos_rate_multiplier": 1.3},
        "model": {"wape_delta": 0.05, "min_units_for_evaluation": 5000},
        "data": {"max_price": 10_000_000, "max_stock": 1_000_000},
    }
    alerts = generate_alerts(df, metrics, thresholds)
    assert len(alerts) >= 1
    assert (alerts["type"] == "SUPPLY").any()
