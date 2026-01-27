from __future__ import annotations

import numpy as np
import pandas as pd


def fillna_with_mean(df: pd.DataFrame, target: str, group: str) -> pd.DataFrame:
    """
    Fill NaN values in `target` with the per-`group` mean of `target`,
    applying floor() to the imputed values.

    Notes:
    - Only NaNs are filled; existing non-null values are not changed.
    - If a group's mean is NaN (group has only NaNs), values remain NaN.
    - Returns a copy of the input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
    target : str
    group : str

    Returns
    -------
    pd.DataFrame
    """
    if target not in df.columns:
        raise KeyError(f"target column '{target}' not found")
    if group not in df.columns:
        raise KeyError(f"group column '{group}' not found")

    result = df.copy()

    means = result.groupby(group)[target].transform("mean")

    filled = result[target].fillna(means)
    result[target] = np.floor(filled)

    return result
