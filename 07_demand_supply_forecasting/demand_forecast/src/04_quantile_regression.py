from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from tqdm import tqdm


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
    # Find the maximum date in the dataset
    max_date = df["day"].max()

    # Calculate the cutoff date for test set
    # test_days days before max_date (inclusive)
    cutoff_date = max_date - pd.Timedelta(days=test_days - 1)

    # Split using >= for test set
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

        Attributes
        ----------
        fitted_models_ : dict
            Dictionary with fitted models for each sku_id.
            Example:
            {
                sku_id_1: {
                    (quantile_1, horizon_1): model_1,
                    (quantile_1, horizon_2): model_2,
                    ...
                },
                sku_id_2: {
                    (quantile_1, horizon_1): model_3,
                    (quantile_1, horizon_2): model_4,
                    ...
                },
                ...
            }

        """
        self.quantiles = quantiles
        self.horizons = horizons
        self.sku_col = "sku_id"
        self.date_col = "day"
        self.features = features
        self.targets = [f"next_{horizon}d" for horizon in self.horizons]

        self.fitted_models_ = {}

    def fit(self, data: pd.DataFrame, verbose: bool = False) -> None:
        """Fit model on data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit on.
        verbose : bool, optional
            Whether to show progress bar, by default False
            Optional to implement, not used in grading.
        """
        # Get unique SKU IDs
        sku_ids = data[self.sku_col].unique()

        # Create iterator with optional progress bar
        sku_iterator = tqdm(sku_ids) if verbose else sku_ids

        for sku_id in sku_iterator:
            # Filter data for current SKU
            sku_data = data[data[self.sku_col] == sku_id].copy()

            # Remove rows with NaN in features or targets
            cols_to_check = self.features + self.targets
            sku_data = sku_data.dropna(subset=cols_to_check)

            # Skip if no data left after dropping NaN
            if len(sku_data) == 0:
                continue

            # Initialize dictionary for this SKU
            self.fitted_models_[sku_id] = {}

            # Extract features
            X = sku_data[self.features].values

            # Train model for each quantile and horizon
            for quantile in self.quantiles:
                for horizon in self.horizons:
                    # Get target column name
                    target_col = f"next_{horizon}d"
                    y = sku_data[target_col].values

                    # Create and fit model
                    model = QuantileRegressor(
                        quantile=quantile,
                        alpha=0.0,  # No regularization
                        solver="highs"
                    )
                    model.fit(X, y)

                    # Store model
                    self.fitted_models_[sku_id][(quantile, horizon)] = model

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict on data.

        Predict 0 values for a new sku_id.

        Parameters
        ----------
        data : pd.DataFrame
            Data to predict on.

        Returns
        -------
        pd.DataFrame
            Predictions.
        """
        # Create output dataframe with sku_id and day columns
        predictions = data[[self.sku_col, self.date_col]].copy()

        # Initialize prediction columns with zeros
        # Order: for each horizon, all quantiles (to match expected output format)
        for horizon in self.horizons:
            for quantile in self.quantiles:
                col_name = f"pred_{horizon}d_q{int(quantile * 100)}"
                predictions[col_name] = 0.0

        # Get unique SKU IDs in prediction data
        sku_ids = data[self.sku_col].unique()

        for sku_id in sku_ids:
            # Check if we have trained models for this SKU
            if sku_id not in self.fitted_models_:
                # Leave predictions as 0 for new SKUs
                continue

            # Get indices for this SKU
            sku_mask = data[self.sku_col] == sku_id
            sku_data = data[sku_mask]

            # Extract features
            X = sku_data[self.features].values

            # Make predictions for each quantile and horizon
            for quantile in self.quantiles:
                for horizon in self.horizons:
                    model_key = (quantile, horizon)

                    if model_key in self.fitted_models_[sku_id]:
                        model = self.fitted_models_[sku_id][model_key]
                        preds = model.predict(X)

                        # Store predictions
                        col_name = f"pred_{horizon}d_q{int(quantile * 100)}"
                        predictions.loc[sku_mask, col_name] = preds

        return predictions


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Calculate the quantile loss between the true and predicted values.

    The quantile loss measures the deviation between the true
        and predicted values at a specific quantile.

    Parameters
    ----------
    y_true : np.ndarray
        The true values.
    y_pred : np.ndarray
        The predicted values.
    quantile : float
        The quantile to calculate the loss for.

    Returns
    -------
    float
        The quantile loss.
    """
    # Calculate the difference
    diff = y_true - y_pred

    # Apply quantile loss formula:
    # L(y, p; q) = q * max(y - p, 0) + (1 - q) * max(p - y, 0)
    loss = np.mean(
        quantile * np.maximum(diff, 0) + (1 - quantile) * np.maximum(-diff, 0)
    )

    return loss
