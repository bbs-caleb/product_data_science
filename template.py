"""Solution for Kaggle AB2: Sequential Forward Selection."""
from typing import Tuple, List

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from tqdm import tqdm


class SequentialForwardSelector:
    """
    Sequential forward selection.
    Algorithm selects features one by one, each time adding the feature that
    improves the model the most.

    Parameters
    ----------
    model: estimator :
        ML model, e.g. LinearRegression
    cv: cross-validation generator :
        cross-validation generator, e.g. KFold, RepeatedKFold
    max_features: int :
        maximum number of features to select
    verbose: int :
        (Default value = 0)
        verbosity level

    Attributes
    ----------
    n_features_: int :
        number of features in the dataset
    selected_features_: List[int] :
        list of selected features, ordered by index
    n_selected_features_: int :
        number of selected features
    """

    def __init__(
            self,
            model,
            cv,
            max_features: int = 10,
            verbose: int = 0,
    ) -> None:
        """Initialize SequentialForwardSelector."""
        self.model = model
        self.cv = cv
        self.max_features = max_features
        self.verbose = verbose
        # Инициализация атрибутов
        self.n_features_ = 0
        self.selected_features_: List[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model.

        Parameters
        ----------
        X: np.ndarray :
            features
        y: np.ndarray :
            target
        """
        self.n_features_ = X.shape[1]
        self.selected_features_ = []
        excluded_features = set(range(self.n_features_))

        # Настройка прогресс-бара, если verbose > 0
        iterations = range(self.max_features)
        if self.verbose > 0:
            iterations = tqdm(iterations, desc="Selecting features")

        # Жадный поиск оптимальных признаков
        for _ in iterations:
            best_score = -np.inf
            best_feature = None

            # Оценка каждого оставшегося признака
            for feature in excluded_features:
                current_subset = self.selected_features_ + [feature]
                X_subset = X[:, current_subset]

                # Кросс-валидация текущего набора
                scores = cross_val_score(
                    self.model, X_subset, y, scoring="r2", cv=self.cv, n_jobs=-1
                )
                mean_score = scores.mean()

                # Обновление лучшего признака
                if mean_score > best_score:
                    best_score = mean_score
                    best_feature = feature

            # Фиксация признака, давшего максимальный прирост метрики
            if best_feature is not None:
                self.selected_features_.append(best_feature)
                excluded_features.remove(best_feature)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce the dataset to selected features.

        Parameters
        ----------
        X: np.ndarray :
            features

        Returns
        -------
        X: np.ndarray :
            reduced dataset
        """
        # Возврат датасета только с отобранными признаками
        return X[:, self.selected_features_]

    @property
    def n_selected_features_(self) -> int:
        """Number of selected features."""
        # Возврат количества отобранных признаков
        return len(self.selected_features_)


def generate_dataset(
        n_samples: int = 10_000,
        n_features: int = 50,
        n_informative: int = 10,
        random_state: int = 42,
) -> Tuple:
    """
    Generate dataset.

    Parameters
    ----------
    n_samples: int :
        (Default value = 10_000)
        number of samples
    n_features: int :
        (Default value = 50)
        number of features
    n_informative: int :
        (Default value = 10)
        number of informative features, other features are noise
    random_state: int :
        (Default value = 42)
        random state for reproducibility

    Returns
    -------
    X: np.ndarray :
        features
    y: np.ndarray :
        target

    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=100,
        random_state=random_state,
        n_informative=n_informative,
        bias=100,
        shuffle=True,
    )
    return X, y


def run() -> None:
    """Run cross-validation and feature selection pipeline."""
    random_state = 42
    n_samples = 10_000
    n_features = 50
    n_informative = 5
    max_features = 10
    n_splits = 3
    n_repeats = 10

    # Генерируем данные
    X, y = generate_dataset(n_samples, n_features, n_informative, random_state)

    # Определяем модель и схему валидации
    model = LinearRegression()
    cv = RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    # Оценка Baseline (все признаки)
    scores = cross_val_score(model, X, y, scoring="r2", cv=cv, n_jobs=-1)
    print(f"Baseline features count: {X.shape[1]}")
    print(f"Baseline R2 score: {scores.mean():.4f}")

    # Запуск SFS отбора
    selector = SequentialForwardSelector(model, cv, max_features, verbose=1)
    selector.fit(X, y)
    X_transformed = selector.transform(X)

    # Оценка после применения SFS
    scores = cross_val_score(model, X_transformed, y, scoring="r2", cv=cv, n_jobs=-1)

    print(f"Features: {selector.selected_features_}")
    print(f"Features count: {selector.n_selected_features_}")
    print(f"Mean R2 score: {scores.mean():.4f}")


if __name__ == "__main__":
    run()
