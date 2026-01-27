import numpy as np
import pandas as pd
import pytest

from src.fillna import fillna_with_mean


def test_fillna_with_mean_floor_rule():
    df = pd.DataFrame(
        {
            "monday": pd.to_datetime(["2020-01-05"] * 10),
            "product_id": [6, 0, 4, 8, 2, 1, 5, 9, 7, 3],
            "category": [3, 0, 2, 0, 2, 5, 4, 0, 5, 5],
            "sales": [35.0, np.nan, 67.0, 4.0, np.nan, np.nan, 32.0, 86.0, 72.0, 5.0],
        }
    )

    out = fillna_with_mean(df, target="sales", group="category")

    # category=0 mean = (4 + 86) / 2 = 45.0
    assert out.loc[out["product_id"] == 0, "sales"].iloc[0] == 45.0

    # category=2 mean = 67.0
    assert out.loc[out["product_id"] == 2, "sales"].iloc[0] == 67.0

    # category=5 mean = (72 + 5) / 2 = 38.5 -> floor => 38.0
    assert out.loc[out["product_id"] == 1, "sales"].iloc[0] == 38.0


def test_does_not_mutate_input():
    df = pd.DataFrame({"g": [1, 1], "y": [np.nan, 10.0]})
    _ = fillna_with_mean(df, target="y", group="g")
    assert pd.isna(df.loc[0, "y"])


def test_missing_columns_raise():
    df = pd.DataFrame({"g": [1], "y": [1.0]})
    with pytest.raises(KeyError):
        fillna_with_mean(df, target="missing", group="g")
    with pytest.raises(KeyError):
        fillna_with_mean(df, target="y", group="missing")
