"""Second-order histogram-based Gradient Boosting with level-wise and leaf-wise growth in pure NumPy."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from ml.algorithms.logistic_regression import sigmoid
from ml.algorithms.tree import BinMapper


class BoostingTree:
    """Regression tree fit on gradients and hessians with flat array storage."""

    def __init__(
        self,
        max_depth: int = 5,
        max_leaves: int = 31,
        min_child_weight: float = 1.0,
        l2_reg: float = 1.0,
        growth_policy: str = "depthwise",  # "depthwise" (XGBoost) or "lossguide" (LightGBM)
    ):
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_child_weight = min_child_weight
        self.l2_reg = l2_reg
        self.growth_policy = growth_policy

        self.feature_: np.ndarray = np.empty(0, dtype=np.int16)
        self.threshold_bin_: np.ndarray = np.empty(0, dtype=np.uint8)
        self.left_child_: np.ndarray = np.empty(0, dtype=np.int32)
        self.right_child_: np.ndarray = np.empty(0, dtype=np.int32)
        self.leaf_value_: np.ndarray = np.empty(0, dtype=np.float64)

    def _calc_leaf_weight(self, g_sum: float, h_sum: float) -> float:
        """Optimal leaf weight: -G / (H + lambda)."""
        return float(-g_sum / (h_sum + self.l2_reg))

    def _calc_gain(self, g_sum: float, h_sum: float) -> float:
        """Split score component: G^2 / (H + lambda)."""
        return float((g_sum ** 2) / (h_sum + self.l2_reg))

    def fit(
        self,
        binned_X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        n_bins: int,
        rng: np.random.Generator,
        colsample_bytree: float = 1.0,
    ) -> "BoostingTree":
        """Fit tree to negative gradients and second-order hessians."""
        n_samples, n_features = binned_X.shape

        if colsample_bytree < 1.0:
            n_sub = max(1, int(colsample_bytree * n_features))
            feature_subset = rng.choice(n_features, size=n_sub, replace=False)
        else:
            feature_subset = np.arange(n_features)

        features_list = []
        thresh_bin_list = []
        left_list = []
        right_list = []
        leaf_val_list = []

        def find_best_split(sample_indices: np.ndarray) -> Tuple[float, int, int]:
            g_sub = g[sample_indices]
            h_sub = h[sample_indices]
            g_total = float(np.sum(g_sub))
            h_total = float(np.sum(h_sub))

            if h_total < self.min_child_weight:
                return 0.0, -1, 0

            parent_score = self._calc_gain(g_total, h_total)
            best_gain = 0.0
            best_feat = -1
            best_bin = 0

            for feat in feature_subset:
                binned_col = binned_X[sample_indices, feat]

                # Accumulate gradients and hessians per bin
                # Using np.bincount with weights
                g_bins = np.bincount(binned_col, weights=g_sub, minlength=n_bins)
                h_bins = np.bincount(binned_col, weights=h_sub, minlength=n_bins)

                # Vectorized cumulative splits
                g_left = np.cumsum(g_bins)[:-1]
                h_left = np.cumsum(h_bins)[:-1]

                g_right = g_total - g_left
                h_right = h_total - h_left

                valid = (h_left >= self.min_child_weight) & (h_right >= self.min_child_weight)
                if not np.any(valid):
                    continue

                gain = 0.5 * (
                    (g_left ** 2) / (h_left + self.l2_reg)
                    + (g_right ** 2) / (h_right + self.l2_reg)
                    - parent_score
                )
                gain = np.where(valid, gain, -1.0)

                max_idx = int(np.argmax(gain))
                if gain[max_idx] > best_gain:
                    best_gain = float(gain[max_idx])
                    best_feat = feat
                    best_bin = max_idx

            return best_gain, best_feat, best_bin

        # Leaf-wise vs Depth-wise building
        root_indices = np.arange(n_samples)
        g_root = float(np.sum(g))
        h_root = float(np.sum(h))

        features_list.append(-1)
        thresh_bin_list.append(0)
        left_list.append(-1)
        right_list.append(-1)
        leaf_val_list.append(self._calc_leaf_weight(g_root, h_root))

        if self.growth_policy == "lossguide":
            # LightGBM-style leaf-wise (best-first) growth
            # Priority queue: list of (gain, node_id, depth, sample_indices, best_feat, best_bin)
            initial_gain, feat, b_bin = find_best_split(root_indices)
            leaves = [(initial_gain, 0, 0, root_indices, feat, b_bin)]

            while len(leaf_val_list) < self.max_leaves * 2 and leaves:
                # Find leaf with highest split gain
                leaves.sort(key=lambda x: x[0], reverse=True)
                gain, node_id, depth, sample_indices, best_feat, best_bin = leaves.pop(0)

                if gain <= 0.0 or best_feat == -1 or depth >= self.max_depth:
                    continue

                left_mask = binned_X[sample_indices, best_feat] <= best_bin
                left_idx = sample_indices[left_mask]
                right_idx = sample_indices[~left_mask]

                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue

                # Turn into decision node
                features_list[node_id] = best_feat
                thresh_bin_list[node_id] = best_bin

                left_id = len(features_list)
                features_list.append(-1)
                thresh_bin_list.append(0)
                left_list.append(-1)
                right_list.append(-1)
                leaf_val_list.append(self._calc_leaf_weight(np.sum(g[left_idx]), np.sum(h[left_idx])))

                right_id = len(features_list)
                features_list.append(-1)
                thresh_bin_list.append(0)
                left_list.append(-1)
                right_list.append(-1)
                leaf_val_list.append(self._calc_leaf_weight(np.sum(g[right_idx]), np.sum(h[right_idx])))

                left_list[node_id] = left_id
                right_list[node_id] = right_id

                # Evaluate children for further splitting
                g_l, f_l, b_l = find_best_split(left_idx)
                g_r, f_r, b_r = find_best_split(right_idx)
                leaves.append((g_l, left_id, depth + 1, left_idx, f_l, b_l))
                leaves.append((g_r, right_id, depth + 1, right_idx, f_r, b_r))

        else:
            # XGBoost-style level-wise depth-first growth
            def build_depthwise(node_id: int, depth: int, sample_indices: np.ndarray):
                if depth >= self.max_depth:
                    return

                gain, best_feat, best_bin = find_best_split(sample_indices)
                if gain <= 0.0 or best_feat == -1:
                    return

                left_mask = binned_X[sample_indices, best_feat] <= best_bin
                left_idx = sample_indices[left_mask]
                right_idx = sample_indices[~left_mask]

                if len(left_idx) == 0 or len(right_idx) == 0:
                    return

                features_list[node_id] = best_feat
                thresh_bin_list[node_id] = best_bin

                left_id = len(features_list)
                features_list.append(-1)
                thresh_bin_list.append(0)
                left_list.append(-1)
                right_list.append(-1)
                leaf_val_list.append(self._calc_leaf_weight(np.sum(g[left_idx]), np.sum(h[left_idx])))

                right_id = len(features_list)
                features_list.append(-1)
                thresh_bin_list.append(0)
                left_list.append(-1)
                right_list.append(-1)
                leaf_val_list.append(self._calc_leaf_weight(np.sum(g[right_idx]), np.sum(h[right_idx])))

                left_list[node_id] = left_id
                right_list[node_id] = right_id

                build_depthwise(left_id, depth + 1, left_idx)
                build_depthwise(right_id, depth + 1, right_idx)

            build_depthwise(0, 0, root_indices)

        self.feature_ = np.array(features_list, dtype=np.int16)
        self.threshold_bin_ = np.array(thresh_bin_list, dtype=np.uint8)
        self.left_child_ = np.array(left_list, dtype=np.int32)
        self.right_child_ = np.array(right_list, dtype=np.int32)
        self.leaf_value_ = np.array(leaf_val_list, dtype=np.float64)

        return self

    def predict(self, binned_X: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of tree predictions."""
        n_samples = binned_X.shape[0]
        current_nodes = np.zeros(n_samples, dtype=np.int32)

        while True:
            feats = self.feature_[current_nodes]
            non_leaf = (feats != -1)
            if not np.any(non_leaf):
                break

            active_idx = np.where(non_leaf)[0]
            active_nodes = current_nodes[active_idx]
            active_feats = feats[active_idx]
            active_thresh = self.threshold_bin_[active_nodes]

            go_left = binned_X[active_idx, active_feats] <= active_thresh
            next_left = self.left_child_[active_nodes]
            next_right = self.right_child_[active_nodes]

            current_nodes[active_idx] = np.where(go_left, next_left, next_right)

        return self.leaf_value_[current_nodes]


class GradientBoostingClassifier:
    """Histogram-based Gradient Boosting with level-wise and leaf-wise options in pure NumPy."""

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        max_leaves: int = 31,
        min_child_weight: float = 1.0,
        l2_reg: float = 1.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        growth_policy: str = "depthwise",  # "depthwise" (XGBoost) or "lossguide" (LightGBM)
        early_stopping_rounds: Optional[int] = 10,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_child_weight = min_child_weight
        self.l2_reg = l2_reg
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.growth_policy = growth_policy
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

        self.trees_: List[BoostingTree] = []
        self.base_score_: float = 0.0
        self.bin_mapper_: Optional[BinMapper] = None
        self.train_losses_: List[float] = []
        self.val_losses_: List[float] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "GradientBoostingClassifier":
        """Fit gradient boosting model with early stopping."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n_samples = X_arr.shape[0]

        rng = np.random.default_rng(self.random_state)

        # Pre-bin features
        self.bin_mapper_ = BinMapper(n_bins=64).fit(X_arr)
        binned_X = self.bin_mapper_.transform(X_arr)

        if X_val is not None:
            binned_X_val = self.bin_mapper_.transform(np.asarray(X_val, dtype=np.float64))
            y_val_arr = np.asarray(y_val, dtype=np.float64).ravel()
        else:
            binned_X_val = None
            y_val_arr = None

        # Base score (prior log-odds)
        p_mean = np.clip(np.mean(y_arr), 1e-4, 1.0 - 1e-4)
        self.base_score_ = float(np.log(p_mean / (1.0 - p_mean)))

        raw_preds = np.full(n_samples, self.base_score_, dtype=np.float64)
        raw_val_preds = (
            np.full(len(y_val_arr), self.base_score_, dtype=np.float64)
            if y_val_arr is not None
            else None
        )

        self.trees_ = []
        self.train_losses_ = []
        self.val_losses_ = []

        best_val_loss = float("inf")
        rounds_no_improve = 0

        for r in range(self.n_estimators):
            # Compute probabilities
            p = sigmoid(raw_preds)

            # Compute negative gradient and hessian for cross-entropy
            g = p - y_arr
            h = np.maximum(p * (1.0 - p), 1e-8)

            # Train loss tracking
            eps = 1e-15
            train_loss = float(-np.mean(y_arr * np.log(np.clip(p, eps, 1.0 - eps)) + (1.0 - y_arr) * np.log(np.clip(1.0 - p, eps, 1.0 - eps))))
            self.train_losses_.append(train_loss)

            # Row subsampling
            if self.subsample < 1.0:
                sub_size = max(1, int(self.subsample * n_samples))
                sub_idx = rng.choice(n_samples, size=sub_size, replace=False)
                train_binned = binned_X[sub_idx]
                train_g = g[sub_idx]
                train_h = h[sub_idx]
            else:
                train_binned = binned_X
                train_g = g
                train_h = h

            tree = BoostingTree(
                max_depth=self.max_depth,
                max_leaves=self.max_leaves,
                min_child_weight=self.min_child_weight,
                l2_reg=self.l2_reg,
                growth_policy=self.growth_policy,
            )
            tree.fit(
                binned_X=train_binned,
                g=train_g,
                h=train_h,
                n_bins=self.bin_mapper_.n_bins,
                rng=rng,
                colsample_bytree=self.colsample_bytree,
            )
            self.trees_.append(tree)

            # Update raw predictions
            tree_preds = tree.predict(binned_X)
            raw_preds += self.learning_rate * tree_preds

            # Validation evaluation and early stopping
            if binned_X_val is not None:
                tree_val_preds = tree.predict(binned_X_val)
                raw_val_preds += self.learning_rate * tree_val_preds
                p_val = sigmoid(raw_val_preds)
                val_loss = float(-np.mean(y_val_arr * np.log(np.clip(p_val, eps, 1.0 - eps)) + (1.0 - y_val_arr) * np.log(np.clip(1.0 - p_val, eps, 1.0 - eps))))
                self.val_losses_.append(val_loss)

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                    if (
                        self.early_stopping_rounds is not None
                        and rounds_no_improve >= self.early_stopping_rounds
                    ):
                        break

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute cumulative raw logits."""
        X_arr = np.asarray(X, dtype=np.float64)
        binned_X = self.bin_mapper_.transform(X_arr)
        raw = np.full(X_arr.shape[0], self.base_score_, dtype=np.float64)
        for tree in self.trees_:
            raw += self.learning_rate * tree.predict(binned_X)
        return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probability."""
        return sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels."""
        return (self.predict_proba(X) >= threshold).astype(np.int64)
