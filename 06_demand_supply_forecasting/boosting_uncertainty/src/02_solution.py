"""Solution for demand forecasting validation task."""
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection._split import _BaseKFold


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error.

    MAPE = mean(|y_true - y_pred| / |y_true|)
    Zeros in y_true are excluded from calculation.
    """
    y_true = np.array(y_true, dtype=np.float64, copy=True)
    y_pred = np.array(y_pred, dtype=np.float64, copy=True)
    mask = y_true != 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error.

    sMAPE = mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|))
    When both y_true and y_pred are 0, that element contributes 0.
    """
    y_true = np.array(y_true, dtype=np.float64, copy=True)
    y_pred = np.array(y_pred, dtype=np.float64, copy=True)
    numerator = 2.0 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    result = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0
    )
    return float(np.mean(result))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Weighted Absolute Percentage Error.

    WAPE = sum(|y_true - y_pred|) / sum(|y_true|)
    """
    y_true = np.array(y_true, dtype=np.float64, copy=True)
    y_pred = np.array(y_pred, dtype=np.float64, copy=True)
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Bias metric.

    Bias = sum(y_pred - y_true) / sum(|y_true|)
    """
    y_true = np.array(y_true, dtype=np.float64, copy=True)
    y_pred = np.array(y_pred, dtype=np.float64, copy=True)
    return float(np.sum(y_pred - y_true) / np.sum(np.abs(y_true)))


class GroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator with non-overlapping groups.

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to groups.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum groups for a single training set.
    test_size : int, default=None
        Number of groups in test set.
    gap : int, default=0
        Number of groups between train and test sets.
    """

    def __init__(self, n_splits=5, max_train_size=None, test_size=None, gap=0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None")

        if hasattr(groups, 'values'):
            groups = groups.values
        groups = np.asarray(groups)

        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        n_folds = self.n_splits + 1
        test_size = self.test_size if self.test_size else n_groups // n_folds

        group_to_indices = {}
        for idx, grp in enumerate(groups):
            if grp not in group_to_indices:
                group_to_indices[grp] = []
            group_to_indices[grp].append(idx)

        group_test_starts = range(
            n_groups - self.n_splits * test_size, n_groups, test_size
        )

        for group_test_start in group_test_starts:
            group_train_end = group_test_start - self.gap

            if self.max_train_size and self.max_train_size < group_train_end:
                group_train_start = group_train_end - self.max_train_size
            else:
                group_train_start = 0

            train_indices = []
            for i in range(group_train_start, group_train_end):
                train_indices.extend(group_to_indices[unique_groups[i]])

            test_indices = []
            for i in range(group_test_start, group_test_start + test_size):
                test_indices.extend(group_to_indices[unique_groups[i]])

            yield np.sort(train_indices), np.sort(test_indices)


def best_model() -> Any:
    """Return the best model for demand forecasting.

    Returns a GradientBoostingRegressor with tuned hyperparameters
    to achieve WAPE in range 20-30%.
    """
    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42,
    )
