"""Module for calculating price elasticity of demand."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def elasticity_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price elasticity for each SKU using R² coefficient.

    For each SKU, fits a linear regression: log(Q+1) = α*P + β
    and returns R² as the elasticity measure.

    Args:
        df: DataFrame with columns 'sku', 'dates', 'price', 'qty'

    Returns:
        DataFrame with columns 'sku' and 'elasticity'
    """
    results = []

    for sku in df["sku"].unique():
        sku_data = df[df["sku"] == sku]

        price = sku_data["price"].values.reshape(-1, 1)
        log_qty = np.log(sku_data["qty"].values + 1)

        model = LinearRegression()
        model.fit(price, log_qty)

        r_squared = model.score(price, log_qty)
        results.append({"sku": sku, "elasticity": r_squared})

    result_df = pd.DataFrame(results)

    return result_df
