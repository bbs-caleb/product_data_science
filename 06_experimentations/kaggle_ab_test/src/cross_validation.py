from typing import Callable, Dict, List
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


def cross_val_score(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    cv: int,
    params_list: List[Dict],
    scoring: Callable,
    random_state: int = 42,
    show_progress: bool = False,
) -> np.ndarray:
    n_models = len(params_list)
    metrics = np.zeros((n_models, cv))

    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)

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
    cv: int,
    model: Callable,
    params_list: List[Dict],
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
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
    baseline_score = avg_scores[0]

    result = []
    for idx in range(1, len(params_list)):
        avg_score = avg_scores[idx]

        if avg_score > baseline_score:
            effect_sign = 1
        elif avg_score < baseline_score:
            effect_sign = -1
        else:
            effect_sign = 0

        result.append({
            "model_index": idx,
            "avg_score": avg_score,
            "effect_sign": effect_sign,
        })

    result.sort(key=lambda x: x["avg_score"], reverse=True)

    return result
