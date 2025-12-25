import numpy as np

def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    """
    Вычисляет ошибку прогноза, сильнее штрафуя за занижение прогноза (недопрогноз),
    чем за завышение (перепрогноз).
    
    Formula: ((y_true - y_pred) / y_pred) ^ 2
    """
    loss = ((y_true - y_pred) / y_pred) ** 2
    return np.mean(loss)
