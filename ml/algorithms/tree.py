"""Histogram-based CART Decision Tree in pure NumPy with flat array representation."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class BinMapper:
    """Pre-computes quantile bin edges to discretize continuous features into integer bins."""

    def __init__(self, n_bins: int = 64):
        self.n_bins = n_bins
        self.bin_edges_: Optional[List[np.ndarray]] = None

    def fit(self, X: np.ndarray) -> "BinMapper":
        """Compute quantile-based bin edges for each feature."""
        X_arr = np.asarray(X, dtype=np.float64)
        n_features = X_arr.shape[1]
        self.bin_edges_ = []

        # Quantile percentiles
        quantiles = np.linspace(0.0, 100.0, self.n_bins + 1)[1:-1]

        for j in range(n_features):
            col = X_arr[:, j]
            edges = np.unique(np.percentile(col, quantiles))
            self.bin_edges_.append(edges)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Discretize continuous float features into uint8 bin indices."""
        if self.bin_edges_ is None:
            raise ValueError("BinMapper has not been fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X_arr.shape
        binned = np.empty((n_samples, n_features), dtype=np.uint8)

        for j in range(n_features):
            edges = self.bin_edges_[j]
            binned[:, j] = np.digitize(X_arr[:, j], edges, right=True).astype(np.uint8)

        return binned

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class DecisionTreeClassifier:
    """Histogram-based CART Decision Tree with flat array representation in pure NumPy."""

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        # Flat tree representation
        self.feature_: np.ndarray = np.empty(0, dtype=np.int16)
        self.threshold_bin_: np.ndarray = np.empty(0, dtype=np.uint8)
        self.threshold_val_: np.ndarray = np.empty(0, dtype=np.float64)
        self.left_child_: np.ndarray = np.empty(0, dtype=np.int32)
        self.right_child_: np.ndarray = np.empty(0, dtype=np.int32)
        self.leaf_value_: np.ndarray = np.empty(0, dtype=np.float64)
        self.n_nodes_: int = 0
        self.bin_mapper_: Optional[BinMapper] = None

    def _gini_gain(
        self,
        binned_col: np.ndarray,
        y: np.ndarray,
        n_bins: int,
    ) -> Tuple[float, int]:
        """Compute the best split for a feature using vectorized histogram binning."""
        n_total = len(y)
        if n_total < self.min_samples_split:
            return 0.0, 0

        # Class counts per bin via bincount
        pos_mask = (y == 1)
        neg_mask = ~pos_mask

        count_pos = np.bincount(binned_col[pos_mask], minlength=n_bins)
        count_neg = np.bincount(binned_col[neg_mask], minlength=n_bins)

        # Cumulative sums evaluate all split thresholds simultaneously
        left_pos = np.cumsum(count_pos)[:-1]
        left_neg = np.cumsum(count_neg)[:-1]
        left_total = left_pos + left_neg

        right_pos = count_pos.sum() - left_pos
        right_neg = count_neg.sum() - left_neg
        right_total = right_pos + right_neg

        # Valid splits satisfying min_samples_leaf
        valid = (left_total >= self.min_samples_leaf) & (right_total >= self.min_samples_leaf)
        if not np.any(valid):
            return 0.0, 0

        # Parent Gini
        parent_p = count_pos.sum() / n_total
        parent_gini = 1.0 - (parent_p ** 2 + (1.0 - parent_p) ** 2)

        # Vectorized Left & Right Gini
        p_left = np.where(left_total > 0, left_pos / np.maximum(1, left_total), 0.0)
        gini_left = 1.0 - (p_left ** 2 + (1.0 - p_left) ** 2)

        p_right = np.where(right_total > 0, right_pos / np.maximum(1, right_total), 0.0)
        gini_right = 1.0 - (p_right ** 2 + (1.0 - p_right) ** 2)

        # Weighted child Gini
        weighted_gini = (left_total / n_total) * gini_left + (right_total / n_total) * gini_right
        gain = np.where(valid, parent_gini - weighted_gini, -1.0)

        best_bin = int(np.argmax(gain))
        best_gain = float(gain[best_bin])

        return best_gain, best_bin

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bin_mapper: Optional[BinMapper] = None,
        binned_X: Optional[np.ndarray] = None,
    ) -> "DecisionTreeClassifier":
        """Build decision tree from training dataset."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64).ravel()
        n_samples, n_features = X_arr.shape

        rng = np.random.default_rng(self.random_state)

        # Handle binning
        if bin_mapper is not None:
            self.bin_mapper_ = bin_mapper
        elif self.bin_mapper_ is None:
            self.bin_mapper_ = BinMapper(n_bins=64).fit(X_arr)

        if binned_X is None:
            binned_X = self.bin_mapper_.transform(X_arr)

        n_bins = self.bin_mapper_.n_bins

        # Resolve max_features
        if self.max_features is None:
            n_sub_features = n_features
        elif self.max_features == "sqrt":
            n_sub_features = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            n_sub_features = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, float):
            n_sub_features = max(1, int(self.max_features * n_features))
        else:
            n_sub_features = int(self.max_features)

        # Tree growth stack: (depth, sample_indices, is_left_child_of_parent, parent_id)
        # Using lists for dynamic node allocation, then converted to flat arrays
        features_list = []
        thresh_bin_list = []
        thresh_val_list = []
        left_list = []
        right_list = []
        leaf_val_list = []

        def build_node(depth: int, sample_indices: np.ndarray) -> int:
            node_id = len(features_list)
            features_list.append(-1)
            thresh_bin_list.append(0)
            thresh_val_list.append(0.0)
            left_list.append(-1)
            right_list.append(-1)
            
            y_subset = y_arr[sample_indices]
            n_node_samples = len(sample_indices)
            p_pos = float(np.mean(y_subset == 1))
            leaf_val_list.append(p_pos)

            # Stopping conditions
            if (
                depth >= self.max_depth
                or n_node_samples < self.min_samples_split
                or p_pos == 0.0
                or p_pos == 1.0
            ):
                return node_id

            # Select candidate features
            if n_sub_features < n_features:
                candidate_features = rng.choice(n_features, size=n_sub_features, replace=False)
            else:
                candidate_features = np.arange(n_features)

            best_gain = 0.0
            best_feat = -1
            best_bin = 0

            # Evaluate candidate features
            for feat in candidate_features:
                binned_col = binned_X[sample_indices, feat]
                gain, bin_idx = self._gini_gain(binned_col, y_subset, n_bins)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_bin = bin_idx

            if best_gain <= 0.0 or best_feat == -1:
                return node_id

            # Partition samples
            left_mask = binned_X[sample_indices, best_feat] <= best_bin
            left_indices = sample_indices[left_mask]
            right_indices = sample_indices[~left_mask]

            if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                return node_id

            # Continuous threshold value for fallback float inference
            edges = self.bin_mapper_.bin_edges_[best_feat]
            val_thresh = float(edges[best_bin]) if best_bin < len(edges) else float(edges[-1])

            # Update current node as internal decision node
            features_list[node_id] = best_feat
            thresh_bin_list[node_id] = best_bin
            thresh_val_list[node_id] = val_thresh

            # Recursively build children
            left_id = build_node(depth + 1, left_indices)
            right_id = build_node(depth + 1, right_indices)

            left_list[node_id] = left_id
            right_list[node_id] = right_id

            return node_id

        root_indices = np.arange(n_samples)
        build_node(0, root_indices)

        # Convert to flat NumPy arrays
        self.n_nodes_ = len(features_list)
        self.feature_ = np.array(features_list, dtype=np.int16)
        self.threshold_bin_ = np.array(thresh_bin_list, dtype=np.uint8)
        self.threshold_val_ = np.array(thresh_val_list, dtype=np.float64)
        self.left_child_ = np.array(left_list, dtype=np.int32)
        self.right_child_ = np.array(right_list, dtype=np.int32)
        self.leaf_value_ = np.array(leaf_val_list, dtype=np.float64)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict positive class probability via vectorized flat array traversal."""
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = X_arr.shape[0]

        if self.bin_mapper_ is not None:
            X_binned = self.bin_mapper_.transform(X_arr)
            use_binned = True
        else:
            use_binned = False

        # Vectorized level-by-level array traversal
        current_nodes = np.zeros(n_samples, dtype=np.int32)
        active = np.ones(n_samples, dtype=bool)

        for _ in range(self.max_depth + 1):
            nodes = current_nodes[active]
            feats = self.feature_[nodes]
            is_leaf = (feats == -1)

            if np.all(is_leaf):
                break

            active_indices = np.where(active)[0]
            non_leaf_mask = ~is_leaf

            if not np.any(non_leaf_mask):
                break

            eval_indices = active_indices[non_leaf_mask]
            eval_nodes = nodes[non_leaf_mask]
            eval_feats = feats[non_leaf_mask]

            if use_binned:
                eval_thresh = self.threshold_bin_[eval_nodes]
                go_left = X_binned[eval_indices, eval_feats] <= eval_thresh
            else:
                eval_thresh_val = self.threshold_val_[eval_nodes]
                go_left = X_arr[eval_indices, eval_feats] <= eval_thresh_val

            next_left = self.left_child_[eval_nodes]
            next_right = self.right_child_[eval_nodes]

            current_nodes[eval_indices] = np.where(go_left, next_left, next_right)
            active[eval_indices[is_leaf[non_leaf_mask]]] = False

        return self.leaf_value_[current_nodes]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels."""
        return (self.predict_proba(X) >= threshold).astype(np.int64)
