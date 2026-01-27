from typing import List
from dataclasses import dataclass

import pandas as pd


@dataclass
class PriceRecommender:
    def recommend_price(self, X: pd.DataFrame, target: float) -> List[bool]:
        def argmax(price):
            return price == price.max()

        mask = X.groupby("sku")["price"].apply(argmax)
        mask = mask.values.tolist()

        return mask

    def gmv(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=X.index)

    def margin(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=X.index)

    def weighted_margin(self, X: pd.DataFrame) -> float:
        return 0.0
