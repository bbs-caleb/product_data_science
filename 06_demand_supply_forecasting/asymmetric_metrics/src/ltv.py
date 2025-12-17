import numpy as np

def ltv_error(y_true: np.array, y_pred: np.array) -> float:
    """
    Вычисляет ошибку LTV, штрафуя за переоценку (Overestimation).
    
    Логика:
    1. Обычная MSPE (деление на y_true) симметрична.
    2. Чтобы штрафовать перепрогноз (когда y_pred > y_true),
       мы умножаем ошибку на коэффициент (y_pred / y_true).
    3. Если y_pred большой -> коэффициент > 1 -> ошибка растет.
       Если y_pred маленький -> коэффициент < 1 -> ошибка уменьшается.
    """
    base_error = ((y_true - y_pred) / y_true) ** 2
    weight = y_pred / y_true
    loss = base_error * weight
    return np.mean(loss)
