
from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "date",
    "sku",
    "category",
    "price",
    "stock",
    "forecast_gmv",
]


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if (df["price"] <= 0).any():
        bad = df.loc[df["price"] <= 0, ["date", "sku", "price"]].head(10)
        raise ValueError(f"Invalid price (<=0) detected. Examples:\n{bad}")

    if (df["stock"] < 0).any():
        bad = df.loc[df["stock"] < 0, ["date", "sku", "stock"]].head(10)
        raise ValueError(f"Invalid stock (<0) detected. Examples:\n{bad}")

    if df[["date", "sku"]].isna().any().any():
        raise ValueError("Nulls detected in key columns (date, sku).")


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["sku"] = out["sku"].astype(int)
    out["price"] = out["price"].astype(float)
    out["stock"] = out["stock"].astype(int)
    out["forecast_gmv"] = out["forecast_gmv"].astype(float)
    if "actual_units" in out.columns:
        out["actual_units"] = out["actual_units"].fillna(0).astype(int)
    return out
