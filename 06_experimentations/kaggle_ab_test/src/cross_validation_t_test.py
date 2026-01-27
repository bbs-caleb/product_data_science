from typing import Callable, Dict, List, Tuple, Union
import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RepeatedKFold


def cross_val_score(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    cv: Union[int, Tuple[int, int]],
    params_list: List[Dict],
    scoring: Callable,
    random_state: int = 42,
    show_progress: bool = False,
) -> np.ndarray:
    if isinstance(cv, tuple):
        n_splits, n_repeats = cv
        kfold = RepeatedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state
        )
        n_folds = n_splits * n_repeats
    else:
        kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        n_folds = cv

    n_models = len(params_list)
    metrics = np.zeros((n_models, n_folds))

    iterator = enumerate(params_list)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="Models")
        except ImportError:
            pass

    for model_idx, params in iterator:
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.set_params(**params)
            model.fit(X_train, np.log1p(y_train))

            y_pred = np.expm1(model.predict(X_val))
            metrics[model_idx, fold_idx] = scoring(y_val, y_pred)

    return metrics


def compare_models(
    cv: Union[int, Tuple[int, int]],
    model: Callable,
    params_list: List[Dict],
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    alpha: float = 0.05,
    show_progress: bool = False,
) -> List[Dict]:
    scores = cross_val_score(
        model=model,
        X=X,
        y=y,
        cv=cv,
        params_list=params_list,
        scoring=r2_score,
        random_state=random_state,
        show_progress=show_progress,
    )

    avg_scores = scores.mean(axis=1)
    baseline_scores = scores[0]
    baseline_avg = avg_scores[0]

    result = []
    for idx in range(1, len(params_list)):
        model_scores = scores[idx]
        avg_score = avg_scores[idx]

        _, p_value = ttest_rel(model_scores, baseline_scores)

        if p_value < alpha:
            if avg_score > baseline_avg:
                effect_sign = 1
            else:
                effect_sign = -1
        else:
            effect_sign = 0

        result.append({
            "model_index": idx,
            "avg_score": avg_score,
            "p_value": p_value,
            "effect_sign": effect_sign,
        })

    result.sort(key=lambda x: x["avg_score"], reverse=True)

    return result
