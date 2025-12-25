from typing import List
from dataclasses import dataclass
import pandas as pd


@dataclass
class PriceRecommender:
    """
    Рекомендатель цен с оптимизацией выручки при соблюдении целевой маржи.
    """

    def gmv(self, X: pd.DataFrame) -> pd.Series:
        """Рассчитывает GMV (выручку) для каждой строки."""
        return X['qty'] * X['price']

    def margin(self, X: pd.DataFrame) -> pd.Series:
        """Рассчитывает маржинальность для каждой строки."""
        return (X['price'] - X['cost']) / X['price']

    def weighted_margin(self, X: pd.DataFrame) -> float:
        """Рассчитывает взвешенную маржинальность."""
        gmv_series = self.gmv(X)
        margin_series = self.margin(X)

        total_gmv = gmv_series.sum()
        if total_gmv == 0:
            return 0.0

        weighted = (margin_series * gmv_series).sum() / total_gmv
        return float(weighted)

    def recommend_price(self, X: pd.DataFrame, target: float) -> List[bool]:
        """
        Рекомендует цены, максимизирующие выручку при соблюдении целевой маржи.

        Алгоритм:
        1. Для каждого SKU выбираем цену с максимальной выручкой
        2. Проверяем, соблюдается ли целевая маржа
        3. Если нет - корректируем цены, начиная с товаров с низкой маржой
        """
        # Добавляем расчетные колонки
        df = X.copy()
        df['gmv'] = self.gmv(df)
        df['margin'] = self.margin(df)

        # Начальное решение: максимальная выручка для каждого SKU
        idx_max_gmv = df.groupby('sku')['gmv'].idxmax()
        mask = pd.Series(False, index=df.index)
        mask.loc[idx_max_gmv] = True

        # Проверяем целевую маржу
        selected_df = df[mask]
        current_margin = self.weighted_margin(selected_df)

        # Если маржа уже соблюдается - возвращаем решение
        if current_margin >= target:
            return mask.tolist()

        # Иначе нужна оптимизация
        # Стратегия: для каждого SKU пробуем более дорогие цены
        skus = df['sku'].unique()

        # Для каждого SKU находим все варианты с сортировкой по цене
        best_mask = mask.copy()
        best_gmv = selected_df['gmv'].sum()

        # Жадный алгоритм: итеративно улучшаем маржу
        for sku in skus:
            sku_data = df[df['sku'] == sku].copy()

            # Сортируем по маржинальности (от большей к меньшей)
            sku_data = sku_data.sort_values('margin', ascending=False)

            for idx in sku_data.index:
                # Пробуем выбрать эту цену
                test_mask = best_mask.copy()
                test_mask[df['sku'] == sku] = False
                test_mask.loc[idx] = True

                test_df = df[test_mask]
                test_margin = self.weighted_margin(test_df)
                test_gmv_sum = test_df['gmv'].sum()

                # Если маржа соблюдается и GMV не хуже - обновляем
                if test_margin >= target:
                    if test_gmv_sum >= best_gmv:
                        best_mask = test_mask
                        best_gmv = test_gmv_sum
                    break

        # Финальная проверка: если всё ещё не соблюдается маржа
        # выбираем комбинацию с максимальной маржой среди всех возможных
        final_df = df[best_mask]
        if self.weighted_margin(final_df) < target:
            # Крайний случай: выбираем максимальную маржу для каждого SKU
            idx_max_margin = df.groupby('sku')['margin'].idxmax()
            best_mask = pd.Series(False, index=df.index)
            best_mask.loc[idx_max_margin] = True

        return best_mask.tolist()


# Тестирование
if __name__ == "__main__":
    # Пример данных
    rows = [
        [1, 40, 40, 100],
        [1, 40, 45, 80],
        [1, 40, 50, 55],
        [2, 100, 200, 300],
        [2, 100, 220, 280],
        [2, 100, 240, 200],
    ]
    df = pd.DataFrame(rows, columns=["sku", "cost", "price", "qty"])

    recommender = PriceRecommender()

    # Тест с низкой целевой маржой (должны выбрать максимальную выручку)
    mask1 = recommender.recommend_price(df, target=0.2)
    print("Target margin: 0.2")
    print("Маска:", mask1)
    print("Выбранные цены:")
    print(df[mask1])
    print(f"GMV: {recommender.gmv(df[mask1]).sum():.2f}")
    print(f"Weighted Margin: {recommender.weighted_margin(df[mask1]):.4f}")
    print()

    # Тест с высокой целевой маржой
    mask2 = recommender.recommend_price(df, target=0.45)
    print("Target margin: 0.45")
    print("Маска:", mask2)
    print("Выбранные цены:")
    print(df[mask2])
    print(f"GMV: {recommender.gmv(df[mask2]).sum():.2f}")
    print(f"Weighted Margin: {recommender.weighted_margin(df[mask2]):.4f}")
