import numpy as np

def smape(y_true: np.array, y_pred: np.array) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).
    Устойчива к нулевым значениям, симметрична.
    Диапазон: 0% - 100% (или 0-200% в зависимости от множителя, здесь шкала 0-1).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2  # Делим на 2 для среднего
    
    # Обработка деления на 0: если и прогноз и факт = 0, ошибка 0
    out = np.zeros_like(denominator, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    
    return float(np.mean(out))

def wape(y_true: np.array, y_pred: np.array) -> float:
    """
    Weighted Average Percentage Error (WAPE).
    Стандарт в ритейле. Показывает отклонение относительно общего объема продаж.
    Не "взрывается" на товарах с низким спросом.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    total_abs_error = np.sum(np.abs(y_true - y_pred))
    total_sales = np.sum(np.abs(y_true))
    
    if total_sales == 0:
        return 0.0
        
    return float(total_abs_error / total_sales)

def mape(y_true: np.array, y_pred: np.array) -> float:
    """
    Mean Absolute Percentage Error (MAPE).
    Классическая метрика, но неустойчива к нулям в y_true.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Избегаем деления на 0
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
        
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

def rmse(y_true: np.array, y_pred: np.array) -> float:
    """Root Mean Squared Error. Сильно штрафует за большие выбросы."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.array, y_pred: np.array) -> float:
    """Mean Absolute Error. Интерпретируемая ошибка в абсолютных единицах."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))
