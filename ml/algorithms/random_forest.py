"""Random Forest Classifier with bootstrap aggregating and out-of-bag evaluation in pure NumPy."""

from typing import Dict, List, Optional, Union
import numpy as np

from ml.algorithms.tree import BinMapper, DecisionTreeClassifier


class RandomForestClassifier:
    """Random Forest ensemble of histogram-based CART trees in pure NumPy."""

    def __init__(
        self,
        n_estimators: int = 25,
        max_depth: int = 8,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        max_features: Union[str, int, float] = "sqrt",
        oob_score: bool = True,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.oob_score = oob_score
        self.random_state = random_state

        self.estimators_: List[DecisionTreeClassifier] = []
        self.bin_mapper_: Optional[BinMapper] = None
        self.oob_score_: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Fit ensemble of trees with bootstrap sampling."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64).ravel()
        n_samples, n_features = X_arr.shape

        rng = np.random.default_rng(self.random_state)

        # Pre-discretize data once for all trees to maximize training speed
        self.bin_mapper_ = BinMapper(n_bins=64).fit(X_arr)
        binned_X = self.bin_mapper_.transform(X_arr)

        self.estimators_ = []

        # Out-of-bag accumulation
        oob_predictions = np.zeros(n_samples, dtype=np.float64)
        oob_counts = np.zeros(n_samples, dtype=np.int32)

        for i in range(self.n_estimators):
            tree_seed = int(rng.integers(0, 1000000))

            # Bootstrap sample
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[boot_idx] = False

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed,
            )

            # Fit using pre-binned data
            tree.fit(
                X=X_arr[boot_idx],
                y=y_arr[boot_idx],
                bin_mapper=self.bin_mapper_,
                binned_X=binned_X[boot_idx],
            )
            self.estimators_.append(tree)

            # Accumulate OOB predictions
            if self.oob_score and np.any(oob_mask):
                oob_preds = tree.predict_proba(X_arr[oob_mask])
                oob_predictions[oob_mask] += oob_preds
                oob_counts[oob_mask] += 1

        # Calculate OOB accuracy
        if self.oob_score:
            valid_oob = oob_counts > 0
            if np.any(valid_oob):
                final_oob_preds = (oob_predictions[valid_oob] / oob_counts[valid_oob]) >= 0.5
                self.oob_score_ = float(np.mean(final_oob_preds == y_arr[valid_oob]))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict average class probability across all trees."""
        if not self.estimators_:
            raise ValueError("RandomForest has not been fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        all_probs = [tree.predict_proba(X_arr) for tree in self.estimators_]
        return np.mean(all_probs, axis=0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels."""
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def save(self, filepath: str) -> None:
        """Save forest parameters and bin edges to a compressed NumPy .npz file."""
        arrays = {
            "n_estimators": np.array([len(self.estimators_)], dtype=np.int32),
            "n_bins": np.array([self.bin_mapper_.n_bins], dtype=np.int32),
        }
        for j, edges in enumerate(self.bin_mapper_.bin_edges_):
            arrays[f"bin_edge_{j}"] = edges

        for i, tree in enumerate(self.estimators_):
            arrays[f"tree_{i}_feature"] = tree.feature_
            arrays[f"tree_{i}_thresh_bin"] = tree.threshold_bin_
            arrays[f"tree_{i}_thresh_val"] = tree.threshold_val_
            arrays[f"tree_{i}_left"] = tree.left_child_
            arrays[f"tree_{i}_right"] = tree.right_child_
            arrays[f"tree_{i}_leaf_val"] = tree.leaf_value_

        np.savez_compressed(filepath, **arrays)
