from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor


def split_train_test(
    df: pd.DataFrame,
    test_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and test sets.

    The last `test_days` days are held out for testing.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        test_days (int): The number of days to include in the test set (default: 30).
            use ">=" sign for df_test

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
        A tuple containing the train and test DataFrames.
    """
    max_date = df["day"].max()
    cutoff_date = max_date - pd.Timedelta(days=test_days - 1)

    df_test = df[df["day"] >= cutoff_date].copy()
    df_train = df[df["day"] < cutoff_date].copy()

    return df_train, df_test


class MultiTargetModel:
    def __init__(
        self,
        features: List[str],
        horizons: List[int] = [7, 14, 21],
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """
        Parameters
        ----------
        features : List[str]
            List of features columns.
        horizons : List[int]
            List of horizons.
        quantiles : List[float]
            List of quantiles.
        """
        self.quantiles = quantiles
        self.horizons = horizons
        self.sku_col = "sku_id"
        self.date_col = "day"
        self.features = features
        self.targets = [f"next_{horizon}d" for horizon in self.horizons]

        self.fitted_models_ = {}

    def fit(self, data: pd.DataFrame, verbose: bool = False) -> None:
        """Fit model on data."""
        sku_ids = data[self.sku_col].unique()

        if verbose:
            try:
                from tqdm import tqdm
                sku_ids = tqdm(sku_ids)
            except ImportError:
                pass

        for sku_id in sku_ids:
            sku_data = data[data[self.sku_col] == sku_id].copy()

            cols_to_check = self.features + self.targets
            sku_data = sku_data.dropna(subset=cols_to_check)

            if len(sku_data) == 0:
                continue

            self.fitted_models_[sku_id] = {}

            X = sku_data[self.features].values

            for quantile in self.quantiles:
                for horizon in self.horizons:
                    target_col = f"next_{horizon}d"
                    y = sku_data[target_col].values

                    model = QuantileRegressor(
                        quantile=quantile,
                        alpha=0.0,
                        solver="highs"
                    )
                    model.fit(X, y)

                    self.fitted_models_[sku_id][(quantile, horizon)] = model

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict on data. Predict 0 values for a new sku_id."""
        predictions = data[[self.sku_col, self.date_col]].copy()

        for horizon in self.horizons:
            for quantile in self.quantiles:
                col_name = f"pred_{horizon}d_q{int(quantile * 100)}"
                predictions[col_name] = 0.0

        sku_ids = data[self.sku_col].unique()

        for sku_id in sku_ids:
            if sku_id not in self.fitted_models_:
                continue

            sku_mask = data[self.sku_col] == sku_id
            sku_data = data[sku_mask]

            X = sku_data[self.features].values

            for quantile in self.quantiles:
                for horizon in self.horizons:
                    model_key = (quantile, horizon)

                    if model_key in self.fitted_models_[sku_id]:
                        model = self.fitted_models_[sku_id][model_key]
                        preds = model.predict(X)

                        col_name = f"pred_{horizon}d_q{int(quantile * 100)}"
                        predictions.loc[sku_mask, col_name] = preds

        return predictions
