# Churn Rate II: Bootstrap

Оценка доверительного интервала ROC-AUC для модели оттока пользователей с помощью бутстрепа.

## Проблема

При оценке модели классификации одной точечной метрики недостаточно. ROC-AUC = 93% не отвечает на вопрос: **в каких пределах будет колебаться метрика на новых данных?**

Метрика — случайная величина, зависящая от выборки. Пользователи меняются, их признаки меняются — нужно понимать границы ожидаемой вариации.

## Решение

**Bootstrap** — метод оценки доверительного интервала без знания аналитического распределения статистики.

Алгоритм:
1. Генерация N псевдовыборок из исходных данных (sampling with replacement)
2. Расчёт ROC-AUC на каждой выборке
3. Определение квантилей для LCB и UCB

![Bootstrap Distribution](notebooks/bootstrap_diagram_.png)

## API

```python
from typing import Tuple
import numpy as np
from sklearn.base import ClassifierMixin

def roc_auc_ci(
    classifier: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    conf: float = 0.95,
    n_bootstraps: int = 10_000,
) -> Tuple[float, float]:
    """Returns confidence bounds of the ROC-AUC"""
    return (lcb, ucb)
```

**Параметры:**
- `classifier` — обученный классификатор sklearn с методом `predict_proba`
- `X`, `y` — тестовая выборка
- `conf` — уровень доверия (default: 0.95)
- `n_bootstraps` — количество бутстреп-выборок

**Возвращает:** `(LCB, UCB)` — нижняя и верхняя границы доверительного интервала
