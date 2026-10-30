"""Vectorized Logistic Regression and StandardScaler in pure NumPy."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class CustomStandardScaler:
    """Feature standardization (z-score scaling) in pure NumPy."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "CustomStandardScaler":
        """Compute the mean and standard deviation for scaling."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X_arr, axis=0)
        std = np.std(X_arr, axis=0)
        # Avoid division by zero for constant features
        self.scale_ = np.where(std == 0.0, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize features by centering and scaling."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("CustomStandardScaler has not been fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        return (X_arr - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, list]:
        """Serialize parameters."""
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else [],
            "scale": self.scale_.tolist() if self.scale_ is not None else [],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, list]) -> "CustomStandardScaler":
        """Deserialize parameters."""
        scaler = cls()
        scaler.mean_ = np.array(data["mean"], dtype=np.float64)
        scaler.scale_ = np.array(data["scale"], dtype=np.float64)
        return scaler


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


class CustomLogisticRegression:
    """Vectorized binary Logistic Regression trained via Mini-Batch Adam in pure NumPy."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        lambda_reg: float = 0.01,
        max_epochs: int = 100,
        batch_size: int = 2048,
        tolerance: float = 1e-5,
        class_weight: Optional[Union[str, Dict[int, float]]] = None,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.class_weight = class_weight
        self.random_state = random_state

        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.loss_history_: List[float] = []
        self.val_loss_history_: List[float] = []

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute sample weights for class imbalance."""
        if self.class_weight is None:
            return np.ones(len(y), dtype=np.float64)

        if self.class_weight == "balanced":
            n_samples = len(y)
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)
            w1 = n_samples / (2.0 * max(1, n_pos))
            w0 = n_samples / (2.0 * max(1, n_neg))
            weights = np.where(y == 1, w1, w0)
            return weights.astype(np.float64)

        if isinstance(self.class_weight, dict):
            w0 = self.class_weight.get(0, 1.0)
            w1 = self.class_weight.get(1, 1.0)
            return np.where(y == 1, w1, w0).astype(np.float64)

        return np.ones(len(y), dtype=np.float64)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute raw linear scores (logits): z = X * w + b."""
        if self.weights_ is None:
            raise ValueError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        return np.dot(X_arr, self.weights_) + self.bias_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities of the positive class (unsafe)."""
        z = self.decision_function(X)
        return sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.int64)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "CustomLogisticRegression":
        """Fit model weights using Mini-Batch Adam optimizer in pure NumPy."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_arr.shape

        rng = np.random.default_rng(self.random_state)
        sample_weights = self._compute_sample_weights(y_arr)

        # Initialize weights
        self.weights_ = np.zeros(n_features, dtype=np.float64)
        self.bias_ = 0.0

        # Adam momentum and RMSprop buffers
        m_w = np.zeros_like(self.weights_)
        v_w = np.zeros_like(self.weights_)
        m_b = 0.0
        v_b = 0.0
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        t = 0

        prev_loss = float("inf")
        self.loss_history_ = []
        self.val_loss_history_ = []

        batch_size = min(self.batch_size, n_samples)

        for epoch in range(self.max_epochs):
            indices = rng.permutation(n_samples)
            X_shuffled = X_arr[indices]
            y_shuffled = y_arr[indices]
            w_shuffled = sample_weights[indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                weights_batch = w_shuffled[start_idx:end_idx]
                m_batch = len(y_batch)

                # Forward pass
                z_batch = np.dot(X_batch, self.weights_) + self.bias_
                y_pred_batch = sigmoid(z_batch)

                # Weighted error
                error = (y_pred_batch - y_batch) * weights_batch

                # Gradients with L2 regularization
                dw = (np.dot(X_batch.T, error) / m_batch) + (self.lambda_reg * self.weights_)
                db = float(np.sum(error) / m_batch)

                # Adam updates
                t += 1
                m_w = beta1 * m_w + (1.0 - beta1) * dw
                v_w = beta2 * v_w + (1.0 - beta2) * (dw ** 2)
                m_b = beta1 * m_b + (1.0 - beta1) * db
                v_b = beta2 * v_b + (1.0 - beta2) * (db ** 2)

                m_w_hat = m_w / (1.0 - beta1 ** t)
                v_w_hat = v_w / (1.0 - beta2 ** t)
                m_b_hat = m_b / (1.0 - beta1 ** t)
                v_b_hat = v_b / (1.0 - beta2 ** t)

                self.weights_ -= (self.learning_rate / (np.sqrt(v_w_hat) + eps)) * m_w_hat
                self.bias_ -= (self.learning_rate / (np.sqrt(v_b_hat) + eps)) * m_b_hat

            # Epoch loss
            train_probs = self.predict_proba(X_arr)
            clipped_probs = np.clip(train_probs, 1e-15, 1.0 - 1e-15)
            epoch_loss = float(
                -np.mean(
                    sample_weights * (y_arr * np.log(clipped_probs) + (1.0 - y_arr) * np.log(1.0 - clipped_probs))
                )
                + 0.5 * self.lambda_reg * np.sum(self.weights_ ** 2)
            )
            self.loss_history_.append(epoch_loss)

            if X_val is not None and y_val is not None:
                val_probs = np.clip(self.predict_proba(X_val), 1e-15, 1.0 - 1e-15)
                val_loss = float(-np.mean(y_val * np.log(val_probs) + (1.0 - y_val) * np.log(1.0 - val_probs)))
                self.val_loss_history_.append(val_loss)

            if abs(prev_loss - epoch_loss) < self.tolerance:
                break
            prev_loss = epoch_loss

        return self

    def save(self, filepath: str) -> None:
        """Save model parameters into a compressed NumPy .npz file."""
        np.savez_compressed(
            filepath,
            weights=self.weights_,
            bias=np.array([self.bias_], dtype=np.float64),
        )

    def load(self, filepath: str) -> "CustomLogisticRegression":
        """Load model parameters from a compressed NumPy .npz file."""
        data = np.load(filepath)
        self.weights_ = data["weights"]
        self.bias_ = float(data["bias"][0])
        return self
