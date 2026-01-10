from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Node:
    n_items: int = 0
    ate: float = 0.0
    split_feat: Optional[int] = None
    split_threshold: float = 0.0
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    is_leaf: bool = True


@dataclass
class UpliftTreeRegressor:
    """
    Uplift Tree Regressor using DeltaDeltaP criterion.

    max_depth : int
        Maximum depth of tree.
    min_samples_leaf : int
        Minimum count of samples in leaf.
    min_samples_leaf_treated : int
        Minimum count of treated samples in leaf.
    min_samples_leaf_control : int
        Minimum count of control samples in leaf.
    """

    max_depth: int = 3
    min_samples_leaf: int = 1000
    min_samples_leaf_treated: int = 300
    min_samples_leaf_control: int = 300

    def __post_init__(self):
        """Initialize the tree root."""
        self.root: Optional[Node] = None

    def _compute_ate(self, y: np.ndarray, treatment: np.ndarray) -> float:
        """
        Compute Average Treatment Effect (ATE) for a node.

        ATE = mean(Y | T=1) - mean(Y | T=0)
        """
        treated_mask = treatment == 1
        control_mask = treatment == 0

        n_treated = np.sum(treated_mask)
        n_control = np.sum(control_mask)

        if n_treated == 0 or n_control == 0:
            return 0.0

        mean_treated = np.sum(y[treated_mask]) / n_treated
        mean_control = np.sum(y[control_mask]) / n_control

        return mean_treated - mean_control

    def _get_threshold_options(self, column_values: np.ndarray) -> np.ndarray:
        """Get threshold options for splitting based on the specified algorithm."""
        unique_values = np.unique(column_values)

        if len(unique_values) > 10:
            percentiles = np.percentile(
                column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97]
            )
        else:
            percentiles = np.percentile(unique_values, [10, 50, 90])

        threshold_options = np.unique(percentiles)
        return threshold_options

    def _check_split_constraints(
        self,
        y_left: np.ndarray,
        treatment_left: np.ndarray,
        y_right: np.ndarray,
        treatment_right: np.ndarray,
    ) -> bool:
        n_left = len(y_left)
        n_right = len(y_right)

        if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
            return False

        n_treated_left = np.sum(treatment_left == 1)
        n_treated_right = np.sum(treatment_right == 1)
        if (
            n_treated_left < self.min_samples_leaf_treated
            or n_treated_right < self.min_samples_leaf_treated
        ):
            return False

        n_control_left = np.sum(treatment_left == 0)
        n_control_right = np.sum(treatment_right == 0)
        if (
            n_control_left < self.min_samples_leaf_control
            or n_control_right < self.min_samples_leaf_control
        ):
            return False

        return True

    def _find_best_split(
        self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best split based on DeltaDeltaP criterion.

        Returns
        -------
        best_feat : int or None
            Index of the best feature to split on.
        best_threshold : float or None
            Best threshold value for the split.
        best_delta : float
            Best DeltaDeltaP value achieved.
        """
        best_feat = None
        best_threshold = None
        best_delta = -np.inf

        n_features = X.shape[1]

        for feat_idx in range(n_features):
            column_values = X[:, feat_idx]
            threshold_options = self._get_threshold_options(column_values)

            for threshold in threshold_options:
                left_mask = column_values <= threshold
                right_mask = ~left_mask

                y_left = y[left_mask]
                treatment_left = treatment[left_mask]
                y_right = y[right_mask]
                treatment_right = treatment[right_mask]

                if not self._check_split_constraints(
                    y_left, treatment_left, y_right, treatment_right
                ):
                    continue

                tau_left = self._compute_ate(y_left, treatment_left)
                tau_right = self._compute_ate(y_right, treatment_right)
                delta = abs(tau_left - tau_right)

                if delta > best_delta:
                    best_delta = delta
                    best_feat = feat_idx
                    best_threshold = threshold

        return best_feat, best_threshold, best_delta

    def _build(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        depth: int,
    ) -> Node:
        """Recursively build the uplift tree."""
        node = Node()
        node.n_items = len(y)
        node.ate = self._compute_ate(y, treatment)

        if depth >= self.max_depth:
            node.is_leaf = True
            return node

        best_feat, best_threshold, best_delta = self._find_best_split(
            X, treatment, y
        )

        if best_feat is None:
            node.is_leaf = True
            return node

        node.is_leaf = False
        node.split_feat = best_feat
        node.split_threshold = best_threshold

        left_mask = X[:, best_feat] <= best_threshold
        right_mask = ~left_mask

        node.left = self._build(
            X[left_mask],
            treatment[left_mask],
            y[left_mask],
            depth + 1,
        )

        node.right = self._build(
            X[right_mask],
            treatment[right_mask],
            y[right_mask],
            depth + 1,
        )

        return node

    def fit(
        self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray
    ) -> "UpliftTreeRegressor":
        """
        Fit the uplift tree model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        treatment : np.ndarray
            Treatment indicator array of shape (n_samples,).
        y : np.ndarray
            Target variable array of shape (n_samples,).

        Returns
        -------
        self : UpliftTreeRegressor
            Fitted model.
        """
        self.root = self._build(X, treatment, y, depth=0)
        return self

    def _predict_single(self, x: np.ndarray, node: Node) -> float:
        """Predict uplift for a single sample."""
        if node.is_leaf:
            return node.ate

        if x[node.split_feat] <= node.split_threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict uplift for samples.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        predictions : np.ndarray
            Predicted uplift values of shape (n_samples,).
        """
        predictions = np.array([self._predict_single(x, self.root) for x in X])
        return predictions
