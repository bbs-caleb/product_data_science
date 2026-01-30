
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class Alert:
    level: str        # INFO / WARN / CRIT
    type: str         # SUPPLY / MODEL / DATA
    message: str
    owner: str        # Supply / Forecast / DWH / Commercial
    action: str


def generate_alerts(
    df_daily: pd.DataFrame,
    metrics: Dict[str, float],
    thresholds: Dict,
) -> pd.DataFrame:
    """Create actionable alerts from governance outputs.

    This is intentionally practical: an alert must have an owner and a next action.
    """
    alerts = []

    # Supply: losses and OOS
    lost_sales_gmv = float(df_daily["lost_sales_gmv"].sum()) if "lost_sales_gmv" in df_daily.columns else 0.0
    oos_rate = float(df_daily["is_stock_constrained"].mean()) if "is_stock_constrained" in df_daily.columns else 0.0

    if lost_sales_gmv >= float(thresholds["supply"]["lost_sales_gmv_week_min"]):
        alerts.append(Alert(
            level="CRIT",
            type="SUPPLY",
            message=f"High lost sales due to OOS: {lost_sales_gmv:,.0f} GMV (period total).",
            owner="Supply/Logistics",
            action="Review top SKUs by lost_sales_gmv; expedite inbound / redistribute / adjust safety stock."
        ))

    # Model: non-OOS quality (if available)
    wape_non_oos = float(metrics.get("wape_non_oos", 0.0))
    if wape_non_oos >= float(thresholds["model"]["wape_delta"]) + float(metrics.get("wape_raw", 0.0)):
        alerts.append(Alert(
            level="WARN",
            type="MODEL",
            message=f"Non-OOS WAPE is elevated: {wape_non_oos:.3f}.",
            owner="Forecast Owner (Planning/DS)",
            action="Check promo factors, seasonality, price elasticity; review recent data changes."
        ))

    # Data: basic sanity already handled in validation; here we can flag extreme values in report
    if (df_daily["price"] > float(thresholds["data"]["max_price"])).any():
        alerts.append(Alert(
            level="WARN",
            type="DATA",
            message="Extreme price values detected (above max_price).",
            owner="DWH/Source Systems",
            action="Validate price feed; check currency/scale issues; correct upstream mapping."
        ))

    return pd.DataFrame([a.__dict__ for a in alerts])
