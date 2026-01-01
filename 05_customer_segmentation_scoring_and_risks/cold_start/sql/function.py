import pandas as pd
import numpy as np


def fillna_with_mean(
    df: pd.DataFrame, target: str, group: str
) -> pd.DataFrame:
    """
    Fills NaN values in the target column with the mean
    value of the group.
    """
    df_result = df.copy()
    means = df_result.groupby(group)[target].transform('mean')
    df_result[target] = df_result[target].fillna(np.floor(means))
    return df_result
