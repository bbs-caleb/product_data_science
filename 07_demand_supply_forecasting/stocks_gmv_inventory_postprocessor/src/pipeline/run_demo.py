
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml

from src.core.feasible_plan import compute_feasible_plan
from src.core.oos_impact import compute_lost_sales
from src.core.metrics import compute_quality_metrics
from src.alerts.rules import generate_alerts
from src.io.utils import coerce_types, validate_input


def main() -> None:
    parser = argparse.ArgumentParser(description="Run demo governance pipeline on a CSV input.")
    parser.add_argument("--input", required=True, help="Path to CSV with sku-day data.")
    parser.add_argument("--outdir", default="report", help="Output directory.")
    parser.add_argument("--thresholds", default="config/thresholds.yaml", help="Thresholds YAML.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = coerce_types(df)
    validate_input(df)

    df1 = compute_feasible_plan(df)
    df2 = compute_lost_sales(df1)

    # Enrich actual GMV if present
    if "actual_units" in df2.columns:
        df2["actual_gmv"] = df2["actual_units"] * df2["price"]

    metrics = compute_quality_metrics(df2)

    with open(args.thresholds, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f)

    alerts = generate_alerts(df2, metrics, thresholds)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df2.to_csv(outdir / "governance_daily.csv", index=False)
    pd.DataFrame([metrics]).to_csv(outdir / "metrics_summary.csv", index=False)
    alerts.to_csv(outdir / "alerts.csv", index=False)

    # Simple management weekly summary (demo)
    weekly = (
        df2.assign(week=pd.to_datetime(df2["date"]).to_period("W").astype(str))
          .groupby(["week", "category"], as_index=False)
          .agg(
              lost_sales_gmv=("lost_sales_gmv", "sum"),
              oos_rate=("is_stock_constrained", "mean"),
              feasible_gmv=("feasible_gmv", "sum"),
              forecast_gmv=("forecast_gmv", "sum"),
          )
          .sort_values(["week", "lost_sales_gmv"], ascending=[False, False])
    )
    weekly.to_csv(outdir / "management_weekly_summary.csv", index=False)


if __name__ == "__main__":
    main()
