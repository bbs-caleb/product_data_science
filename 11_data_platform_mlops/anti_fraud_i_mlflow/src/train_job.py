"""Anti-fraud model training job with MLflow logging."""
import os
from typing import Any, List, Tuple

import fire
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay

IDENTIFIER = f'antifraud-{os.environ.get("KCHECKER_USER_USERNAME", "default")}'
TRACKING_URI = os.environ.get("TRACKING_URI")


def recall_at_precision(
    true_labels: np.ndarray,
    pred_scores: np.ndarray,
    min_precision: float = 0.95,
) -> float:
    """Compute recall at precision.

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores
        min_precision (float, optional): Min precision for recall. Defaults to 0.95.

    Returns:
        float: Metric value
    """
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    valid_indices = precision >= min_precision
    if not np.any(valid_indices):
        return 0.0
    metric = float(np.max(recall[valid_indices]))
    return metric


def recall_at_specificity(
    true_labels: np.ndarray,
    pred_scores: np.ndarray,
    min_specificity: float = 0.95,
) -> float:
    """Compute recall at specificity.

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores
        min_specificity (float, optional): Min specificity for recall. Defaults to 0.95.

    Returns:
        float: Metric value
    """
    fpr, tpr, _ = roc_curve(true_labels, pred_scores)
    max_fpr = 1 - min_specificity
    valid_indices = fpr <= max_fpr
    if not np.any(valid_indices):
        return 0.0
    metric = float(np.max(tpr[valid_indices]))
    return metric


def curves(true_labels: np.ndarray, pred_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return PR and ROC curves as numpy arrays.

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores

    Returns:
        Tuple[np.ndarray, np.ndarray]: PR and ROC curves
    """
    def fig2numpy(fig: Any) -> np.ndarray:
        fig.canvas.draw()
        img = fig.canvas.buffer_rgba()
        img = np.asarray(img)
        return img

    pr_display = PrecisionRecallDisplay.from_predictions(true_labels, pred_scores)
    pr_curve = fig2numpy(pr_display.figure_)

    roc_display = RocCurveDisplay.from_predictions(true_labels, pred_scores)
    roc_curve_img = fig2numpy(roc_display.figure_)

    return pr_curve, roc_curve_img


def job(
    train_path: str = "",
    test_path: str = "",
    target: str = "target",
    model_type: str = "IsolationForest",
    n_estimators: int = 100,
    max_samples: str = "auto",
    contamination: float = 0.1,
    nu: float = 0.1,
    n_neighbors: int = 20,
):
    """Model training job.

    Args:
        train_path (str): Train dataset path
        test_path (str): Test dataset path
        target (str): Target column name
        model_type (str): Model type (IsolationForest, LocalOutlierFactor, OneClassSVM)
        n_estimators (int): Number of estimators for IsolationForest
        max_samples (str): Max samples for IsolationForest
        contamination (float): Contamination for IsolationForest/LOF
        nu (float): Nu parameter for OneClassSVM
        n_neighbors (int): Number of neighbors for LocalOutlierFactor
    """
    # Настройка MLflow
    if TRACKING_URI:
        mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(IDENTIFIER)
    mlflow.start_run()

    # Загрузка данных
    train_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)

    # Определение признаков
    features: List[str] = [col for col in train_dataset.columns if col != target]

    # Подготовка данных
    X_train = train_dataset[features]
    X_test = test_dataset[features]
    test_targets = test_dataset[target].values

    # Создание модели в зависимости от типа
    if model_type == "IsolationForest":
        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=42
        )
        model.fit(X_train)
        pred_scores = -model.score_samples(X_test)
    elif model_type == "LocalOutlierFactor":
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True
        )
        model.fit(X_train)
        pred_scores = -model.score_samples(X_test)
    elif model_type == "OneClassSVM":
        model = OneClassSVM(nu=nu, kernel="rbf", gamma="auto")
        model.fit(X_train)
        pred_scores = -model.score_samples(X_test)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Вычисление метрик
    roc_auc = roc_auc_score(test_targets, pred_scores)
    recall_precision_95 = recall_at_precision(test_targets, pred_scores, min_precision=0.95)
    recall_specificity_95 = recall_at_specificity(test_targets, pred_scores, min_specificity=0.95)

    # Логирование тегов
    mlflow.set_tags({
        "task_type": "anti-fraud",
        "framework": "sklearn"
    })

    # Логирование параметров датасета
    mlflow.log_param("features", features)
    mlflow.log_param("target", target)

    # Логирование параметров модели
    mlflow.log_param("model_type", model.__class__.__name__)

    if model_type == "IsolationForest":
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_samples", max_samples)
        mlflow.log_param("contamination", contamination)
    elif model_type == "LocalOutlierFactor":
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("contamination", contamination)
    elif model_type == "OneClassSVM":
        mlflow.log_param("nu", nu)

    # Логирование метрик
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("recall_precision_95", recall_precision_95)
    mlflow.log_metric("recall_specificity_95", recall_specificity_95)

    # Логирование артефактов - датасеты
    mlflow.log_artifact(train_path, "data")
    mlflow.log_artifact(test_path, "data")

    # Логирование артефактов - графики
    pr_curve, roc_curve_img = curves(test_targets, pred_scores)
    mlflow.log_image(pr_curve, "metrics/pr.png")
    mlflow.log_image(roc_curve_img, "metrics/roc.png")

    # Логирование модели
    mlflow.sklearn.log_model(
        model,
        artifact_path=IDENTIFIER,
        registered_model_name=IDENTIFIER
    )

    print(f"Model: {model_type}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Recall@Precision95: {recall_precision_95:.4f}")
    print(f"Recall@Specificity95: {recall_specificity_95:.4f}")

    mlflow.end_run()


if __name__ == "__main__":
    fire.Fire(job)
