"""Probability calibration using Platt Scaling and Isotonic Regression (PAVA) in pure NumPy."""

from typing import Dict, Tuple, Union
import numpy as np


class PlattCalibrator:
    """Platt Scaling: fits a sigmoid mapping P(y=1|z) = 1 / (1 + exp(-(a*z + b))) using Newton's method."""

    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.a_: float = 1.0
        self.b_: float = 0.0

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        """Fit parameters a and b using Newton-Raphson optimization in pure NumPy."""
        scores_arr = np.asarray(scores, dtype=np.float64).ravel()
        y_true = np.asarray(y, dtype=np.float64).ravel()

        # Target smoothing recommended by Platt (1999) to avoid overfitting
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        t_pos = (n_pos + 1.0) / (n_pos + 2.0)
        t_neg = 1.0 / (n_neg + 2.0)
        targets = np.where(y_true == 1, t_pos, t_neg)

        # Initial parameters
        a = 1.0
        b = float(np.log((n_pos + 1.0) / (n_neg + 1.0)))

        for _ in range(self.max_iter):
            # Compute probabilities
            z = a * scores_arr + b
            z_clipped = np.clip(z, -500.0, 500.0)
            p = 1.0 / (1.0 + np.exp(-z_clipped))

            # Gradient of negative log-likelihood: p - target
            diff = p - targets
            ga = float(np.sum(diff * scores_arr))
            gb = float(np.sum(diff))

            # Hessian components
            w = np.maximum(p * (1.0 - p), 1e-12)
            h11 = float(np.sum(w * (scores_arr ** 2)))
            h22 = float(np.sum(w))
            h12 = float(np.sum(w * scores_arr))

            det = h11 * h22 - (h12 ** 2)
            if det > 1e-12:
                step_a = (h22 * ga - h12 * gb) / det
                step_b = (-h12 * ga + h11 * gb) / det
                a -= step_a
                b -= step_b
            else:
                a -= 0.01 * ga
                b -= 0.01 * gb
                step_a, step_b = 0.0, 0.0

            if abs(step_a) < self.tol and abs(step_b) < self.tol:
                break

        self.a_ = a
        self.b_ = b
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Transform raw model scores into calibrated probabilities."""
        scores_arr = np.asarray(scores, dtype=np.float64)
        z = scores_arr * self.a_ + self.b_
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def to_dict(self) -> Dict[str, float]:
        return {"a": float(self.a_), "b": float(self.b_)}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "PlattCalibrator":
        cal = cls()
        cal.a_ = float(data.get("a", data.get("A", 1.0)))
        cal.b_ = float(data.get("b", data.get("B", 0.0)))
        return cal


class IsotonicCalibrator:
    """Non-parametric isotonic regression via Pool Adjacent Violators Algorithm (PAVA)."""

    def __init__(self):
        self.x_thresholds_: Optional[np.ndarray] = None
        self.y_values_: Optional[np.ndarray] = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        """Fit monotonically non-decreasing step function via PAVA in pure NumPy."""
        z = np.asarray(scores, dtype=np.float64).ravel()
        y_true = np.asarray(y, dtype=np.float64).ravel()

        # Sort by scores ascending
        order = np.argsort(z)
        z_sorted = z[order]
        y_sorted = y_true[order]

        # PAVA algorithm
        # Each block: [weight, sum_y, x_min, x_max]
        blocks = []
        for xi, yi in zip(z_sorted, y_sorted):
            blocks.append([1.0, float(yi), float(xi), float(xi)])
            while len(blocks) >= 2:
                prev_val = blocks[-2][1] / blocks[-2][0]
                curr_val = blocks[-1][1] / blocks[-1][0]
                if prev_val >= curr_val:
                    # Pool adjacent violators
                    b2 = blocks.pop()
                    b1 = blocks.pop()
                    new_w = b1[0] + b2[0]
                    new_sum = b1[1] + b2[1]
                    new_min = b1[2]
                    new_max = b2[3]
                    blocks.append([new_w, new_sum, new_min, new_max])
                else:
                    break

        x_thresh = []
        y_vals = []
        for b in blocks:
            x_thresh.append((b[2] + b[3]) / 2.0)
            y_vals.append(b[1] / b[0])

        self.x_thresholds_ = np.array(x_thresh, dtype=np.float64)
        self.y_values_ = np.clip(np.array(y_vals, dtype=np.float64), 0.0, 1.0)
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Interpolate calibrated probability along the monotonic curve."""
        if self.x_thresholds_ is None or self.y_values_ is None:
            raise ValueError("IsotonicCalibrator is not fitted yet.")
        z = np.asarray(scores, dtype=np.float64)
        return np.interp(z, self.x_thresholds_, self.y_values_, left=self.y_values_[0], right=self.y_values_[-1])
