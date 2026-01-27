"""Solution for demand forecasting validation task."""
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection._split import _BaseKFold


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error.
    
    MAPE = mean(|y_true - y_pred| / |y_true|) * 100
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Избегаем деления на ноль
    mask = y_true != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error.
    
    sMAPE = mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|)) * 100
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    numerator = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    # Избегаем деления на ноль
    mask = denominator != 0
    return np.mean(2 * numerator[mask] / denominator[mask]) * 100


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error.
    
    WAPE = sum(|y_true - y_pred|) / sum(|y_true|) * 100
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Bias metric.
    
    Bias = sum(y_pred - y_true) / sum(y_true) * 100
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sum(y_pred - y_true) / np.sum(y_true) * 100


class GroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum groups for a single training set.
    test_size : int, default=None
        Number of groups in test
    gap : int, default=0
        Number of groups between train and test sets
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
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
            
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None")

        # Преобразуем в numpy array
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(groups, 'values'):
            groups = groups.values
        groups = np.asarray(groups)
        n_samples = len(groups)

        # Получаем уникальные группы в порядке их первого появления
        _, first_occurrence_idx = np.unique(groups, return_index=True)
        sorted_idx = np.argsort(first_occurrence_idx)
        unique_groups = groups[first_occurrence_idx[sorted_idx]]
        n_groups = len(unique_groups)

        # Определяем test_size (по умолчанию 1 группа)
        test_size = self.test_size if self.test_size is not None else 1

        # Создаём словарь: группа -> список индексов
        group_to_indices = {g: [] for g in unique_groups}
        for idx, g in enumerate(groups):
            group_to_indices[g].append(idx)

        # Генерируем сплиты
        for i in range(self.n_splits):
            # Позиции групп для test
            test_end = n_groups - (self.n_splits - 1 - i) * test_size
            test_start = test_end - test_size

            # Позиции групп для train (с учётом gap)
            train_end = test_start - self.gap
            train_start = 0
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)

            # Собираем индексы для train
            train_indices = []
            for j in range(train_start, train_end):
                train_indices.extend(group_to_indices[unique_groups[j]])

            # Собираем индексы для test
            test_indices = []
            for j in range(test_start, test_end):
                test_indices.extend(group_to_indices[unique_groups[j]])

            yield np.array(train_indices), np.array(test_indices)


def best_model() -> Any:
    """Return the best model for demand forecasting.
    
    Returns a GradientBoostingRegressor with tuned hyperparameters
    to achieve WAPE in range 20-30%.
    """
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )
    return model
