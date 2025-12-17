from typing import Dict, Tuple, Optional
import pandas as pd

# --- КОНФИГУРАЦИЯ ---
FEATURES = {
    "qty_7d_avg": ("qty", 7, "avg", None),
    "qty_7d_q10": ("qty", 7, "quantile", 10),
    "qty_7d_q50": ("qty", 7, "quantile", 50),
    "qty_7d_q90": ("qty", 7, "quantile", 90),
    "qty_14d_avg": ("qty", 14, "avg", None),
    "qty_14d_q10": ("qty", 14, "quantile", 10),
    "qty_14d_q50": ("qty", 14, "quantile", 50),
    "qty_14d_q90": ("qty", 14, "quantile", 90),
    "qty_21d_avg": ("qty", 21, "avg", None),
    "qty_21d_q10": ("qty", 21, "quantile", 10),
    "qty_21d_q50": ("qty", 21, "quantile", 50),
    "qty_21d_q90": ("qty", 21, "quantile", 90),
}

TARGETS = {
    "next_7d": ("qty", 7),
    "next_14d": ("qty", 14),
    "next_21d": ("qty", 21),
}
# --- ФУНКЦИИ ---

def add_features(
    df: pd.DataFrame,
    features: Dict[str, Tuple[str, int, str, Optional[int]]],
) -> None:
    """
    Добавляет признаки на основе прошлых данных (включая текущий день).
    """
    grouped = df.groupby("sku_id")
    for feature_name, (agg_col, days, agg_func, quantile) in features.items():
        if agg_func == "quantile":
            df[feature_name] = (
                grouped[agg_col]
                .rolling(window=days)
                .quantile(quantile / 100)
                .reset_index(level=0, drop=True)
            )
        elif agg_func == "avg":
            df[feature_name] = (
                grouped[agg_col]
                .rolling(window=days)
                .mean()
                .reset_index(level=0, drop=True)
            )
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")

def add_targets(df: pd.DataFrame, targets: Dict[str, Tuple[str, int]]) -> None:
    """
    Добавляет таргеты: сумма продаж за СЛЕДУЮЩИЕ N дней (не включая текущий).
    """
    for target_name, (agg_col, days) in targets.items():
        # Использование days=days в lambda решает проблему W0640 (Cell variable)
        df[target_name] = df.groupby("sku_id")[agg_col].transform(
            lambda x, days=days: x.shift(-1)
                                 .rolling(window=days)
                                 .sum()
                                 .shift(-(days - 1))
        )
# --- БЛОК ВЫПОЛНЕНИЯ ---
if __name__ == "__main__":
    # 1. Загружаем данные
    df = pd.read_csv('sales.csv')
    # 2. Подготовка: конвертация даты и СОРТИРОВКА (критично)
    df['day'] = pd.to_datetime(df['day'])
    df = df.sort_values(['sku_id', 'day'])
    # 3. Генерация признаков и таргетов
    add_features(df, FEATURES)
    add_targets(df, TARGETS)
    # 4. Сохранение результата
    df.to_csv('features.csv', index=False)
