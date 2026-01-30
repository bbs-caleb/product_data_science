from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd


def add_features(
    df: pd.DataFrame,
    features: dict[str, Tuple[str, int, str, Optional[int]]],
) -> pd.DataFrame:
    """
    Add features to the DataFrame based on the features dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to add features to.
    features : dict[str, Tuple[str, int, str, Optional[int]]]
        A dictionary where keys are feature names and values are tuples of
        (column_name, window_size, aggregation_type, quantile_value).
        aggregation_type can be 'avg' or 'quantile'.
        quantile_value is required when aggregation_type is 'quantile' (0-100).

    Returns
    -------
    pd.DataFrame
        The DataFrame with the new features added.
    """
    df = df.copy()

    for feature_name, (col, window, agg_type, q_value) in features.items():
        if agg_type == "avg":
            df[feature_name] = (
                df.groupby("sku_id")[col]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )
        elif agg_type == "quantile":
            quantile = q_value / 100.0
            df[feature_name] = (
                df.groupby("sku_id")[col]
                .transform(
                    lambda x: x.rolling(window=window, min_periods=1).quantile(quantile)
                )
            )

    return df


def add_targets(
    df: pd.DataFrame,
    targets: dict[str, Tuple[str, int]],
) -> pd.DataFrame:
    """
    Add target columns to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to add targets to.
    targets : dict[str, Tuple[str, int]]
        A dictionary where keys are target names and values are tuples of
        (column_name, horizon_days).

    Returns
    -------
    pd.DataFrame
        The DataFrame with target columns added.
    """
    df = df.copy()

    for target_name, (col, horizon) in targets.items():
        df[target_name] = (
            df.groupby("sku_id")[col]
            .transform(
                lambda x: x.shift(-1).rolling(window=horizon, min_periods=horizon).sum()
            )
        )

    return df
