"""Weighted Soft-Voting Ensemble in pure NumPy."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from ml.algorithms.calibration import PlattCalibrator


class WeightedEnsemble:
    """Ensemble combining predictions of multiple base estimators via weighted soft-voting."""

    def __init__(
        self,
        estimators: Optional[List[Tuple[str, object]]] = None,
        weights: Optional[List[float]] = None,
        calibrator: Optional[PlattCalibrator] = None,
    ):
        self.estimators = estimators or []
        self.weights = weights
        self.calibrator = calibrator

        # Normalized weights
        if weights is not None:
            w_arr = np.array(weights, dtype=np.float64)
            self.weights_ = w_arr / np.sum(w_arr)
        else:
            self.weights_ = None

    def fit_weights(self, X_val: np.ndarray, y_val: np.ndarray, n_grid: int = 21) -> "WeightedEnsemble":
        """Optimize ensemble weights on validation predictions."""
        n_est = len(self.estimators)
        if n_est == 0:
            return self

        val_probs = [est.predict_proba(X_val) for _, est in self.estimators]
        val_probs_arr = np.column_stack(val_probs)  # Shape (N, n_est)
        y_val_arr = np.asarray(y_val, dtype=np.float64).ravel()

        if n_est == 2:
            # 1D line search
            best_w = np.array([0.5, 0.5])
            best_loss = float("inf")
            for w0 in np.linspace(0.0, 1.0, n_grid):
                w = np.array([w0, 1.0 - w0])
                blend = np.dot(val_probs_arr, w)
                blend_clipped = np.clip(blend, 1e-15, 1.0 - 1e-15)
                loss = -np.mean(y_val_arr * np.log(blend_clipped) + (1.0 - y_val_arr) * np.log(1.0 - blend_clipped))
                if loss < best_loss:
                    best_loss = loss
                    best_w = w
            self.weights_ = best_w
        elif n_est == 3:
            # 2D simplex search
            best_w = np.ones(3) / 3.0
            best_loss = float("inf")
            for w0 in np.linspace(0.0, 1.0, 11):
                for w1 in np.linspace(0.0, 1.0 - w0, 11):
                    w2 = 1.0 - w0 - w1
                    w = np.array([w0, w1, w2])
                    blend = np.dot(val_probs_arr, w)
                    blend_clipped = np.clip(blend, 1e-15, 1.0 - 1e-15)
                    loss = -np.mean(y_val_arr * np.log(blend_clipped) + (1.0 - y_val_arr) * np.log(1.0 - blend_clipped))
                    if loss < best_loss:
                        best_loss = loss
                        best_w = w
            self.weights_ = best_w
        else:
            self.weights_ = np.ones(n_est) / float(n_est)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute weighted average probability across all estimators."""
        if not self.estimators:
            raise ValueError("No estimators configured in ensemble.")

        if self.weights_ is None:
            self.weights_ = np.ones(len(self.estimators), dtype=np.float64) / len(self.estimators)

        probs = [est.predict_proba(X) for _, est in self.estimators]
        probs_mat = np.column_stack(probs)
        raw_blend = np.dot(probs_mat, self.weights_)

        if self.calibrator is not None:
            # Calibrate raw ensemble probabilities
            eps = 1e-15
            clipped = np.clip(raw_blend, eps, 1.0 - eps)
            logits = np.log(clipped / (1.0 - clipped))
            return self.calibrator.predict_proba(logits)

        return raw_blend

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels."""
        return (self.predict_proba(X) >= threshold).astype(np.int64)
