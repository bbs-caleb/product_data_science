"""EASE (Embarrassingly Shallow Autoencoders) recommender system."""

import numpy as np
import pandas as pd


class EASE:
    """Linear recommender based on item-item weight matrix with L2 regularization."""

    def __init__(self, lambda_reg):
        """Initialize EASE model.

        Args:
            lambda_reg (float): L2 regularization strength.
        """
        self.lambda_reg = lambda_reg
        self.W = None  # pylint: disable=C0103

    def fit(self, basket_item_matrix):
        """Train weight matrix W from basket-item interactions.

        Args:
            basket_item_matrix (numpy.ndarray): Binary interaction matrix (baskets x items).
        """
        if not isinstance(basket_item_matrix, np.ndarray):
            raise ValueError("Матрица basket_item_matrix должна быть numpy.ndarray")
        if basket_item_matrix.ndim != 2:
            raise ValueError("Матрица basket_item_matrix должна быть двумерной")

        # Gram matrix: item-item co-occurrence similarity
        gram = (basket_item_matrix.T @ basket_item_matrix).astype(np.float64)

        # Add L2 regularization to diagonal
        diag_idx = np.diag_indices(gram.shape[0])
        gram[diag_idx] += self.lambda_reg

        # Closed-form EASE solution: W = I - P * diag(1/diag(P))
        inv_gram = np.linalg.inv(gram)
        self.W = -inv_gram / np.diag(inv_gram)[:, None]

        # Zero out self-connections
        self.W[diag_idx] = 0

    def predict(self, basket_item_matrix):
        """Compute predicted scores for all baskets.

        Args:
            basket_item_matrix (numpy.ndarray): Binary interaction matrix (baskets x items).

        Returns:
            numpy.ndarray: Score matrix (baskets x items).
        """
        if self.W is None:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")

        # Score = interaction vector multiplied by weight matrix
        return basket_item_matrix @ self.W

    def recommend(self, basket_item_matrix, basket_index, top_k=5):
        """Get top-k recommendations for a specific basket.

        Args:
            basket_item_matrix (numpy.ndarray): Binary interaction matrix (baskets x items).
            basket_index (int): Row index of the target basket.
            top_k (int): Number of items to recommend.

        Returns:
            list: Indices of recommended items.
        """
        if basket_index < 0 or basket_index >= basket_item_matrix.shape[0]:
            raise ValueError(
                f"Индекс корзины должен быть от 0 до {basket_item_matrix.shape[0] - 1}"
            )

        # Predict scores for target basket
        scores = self.predict(basket_item_matrix)[basket_index]

        # Mask items already in the basket
        scores[basket_item_matrix[basket_index] > 0] = -np.inf

        # Return top-k highest scoring item indices
        recommended_indices = np.argsort(scores)[::-1][:top_k].tolist()
        return recommended_indices


def test():
    """Smoke test on the provided dataset."""
    data = pd.read_csv("/mnt/user-data/uploads/basket_item_dataset.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    matrix = data.pivot_table(
        index="basket_id", columns="item_id",
        values="timestamp", aggfunc="count", fill_value=0
    ).values

    model = EASE(lambda_reg=1.0)
    model.fit(matrix)

    predictions = model.predict(matrix)
    print(f"Interaction matrix shape: {matrix.shape}")
    print(f"Weight matrix shape:      {model.W.shape}")
    print(f"Predictions matrix shape: {predictions.shape}")
    print(f"W diagonal (should be 0): {np.diag(model.W).sum():.6f}")

    recs = model.recommend(matrix, basket_index=0, top_k=5)
    print(f"Top-5 recommendations for basket 0: {recs}")


if __name__ == "__main__":
    test()
