"""From-scratch classification metrics and evaluation engine in pure NumPy."""

from typing import Dict, Tuple, Union
import numpy as np


def _to_numpy(arr: Union[np.ndarray, list]) -> np.ndarray:
    """Ensure input is a 1D NumPy array."""
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return arr.ravel()


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculate binary confusion matrix [[TN, FP], [FN, TP]]."""
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate classification accuracy."""
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)
    if len(y_t) == 0:
        return 0.0
    return float(np.mean(y_t == y_p))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate precision = TP / (TP + FP)."""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate recall = TP / (TP + FN)."""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate F1 score = 2 * (P * R) / (P + R)."""
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    denom = p + r
    return float(2 * p * r / denom) if denom > 0 else 0.0


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate false positive rate = FP / (FP + TN)."""
    cm = confusion_matrix(y_true, y_pred)
    fp = cm[0, 1]
    tn = cm[0, 0]
    denom = fp + tn
    return float(fp / denom) if denom > 0 else 0.0


def trapezoidal_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Area Under the Curve using the trapezoidal numerical integration rule."""
    x = _to_numpy(x)
    y = _to_numpy(y)
    if len(x) < 2:
        return 0.0
    # Trapezoid rule: sum 0.5 * (x[i+1] - x[i]) * (y[i+1] + y[i])
    dx = np.diff(x)
    avg_y = (y[:-1] + y[1:]) / 2.0
    return float(np.sum(dx * avg_y))


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Receiver Operating Characteristic (ROC) curve points.

    Returns:
        (fpr, tpr, thresholds) arrays.
    """
    y_t = _to_numpy(y_true)
    y_s = _to_numpy(y_score)

    n_pos = np.sum(y_t == 1)
    n_neg = np.sum(y_t == 0)

    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    # Sort scores descending
    desc_order = np.argsort(y_s)[::-1]
    y_s_sorted = y_s[desc_order]
    y_t_sorted = y_t[desc_order]

    # Find distinct thresholds
    distinct_idx = np.where(np.diff(y_s_sorted))[0]
    threshold_idxs = np.concatenate([distinct_idx, [len(y_s_sorted) - 1]])

    tps = np.cumsum(y_t_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_t_sorted == 0)[threshold_idxs]

    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    thresholds = np.concatenate([[y_s_sorted[0] + 1.0], y_s_sorted[threshold_idxs]])

    return fpr, tpr, thresholds


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the ROC Curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return trapezoidal_auc(fpr, tpr)


def precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Precision-Recall curve points.

    Returns:
        (precision, recall, thresholds) arrays.
    """
    y_t = _to_numpy(y_true)
    y_s = _to_numpy(y_score)

    n_pos = np.sum(y_t == 1)
    if n_pos == 0:
        return np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0])

    desc_order = np.argsort(y_s)[::-1]
    y_s_sorted = y_s[desc_order]
    y_t_sorted = y_t[desc_order]

    distinct_idx = np.where(np.diff(y_s_sorted))[0]
    threshold_idxs = np.concatenate([distinct_idx, [len(y_s_sorted) - 1]])

    tps = np.cumsum(y_t_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_t_sorted == 0)[threshold_idxs]

    precision = tps / (tps + fps)
    recall = tps / n_pos
    thresholds = y_s_sorted[threshold_idxs]

    # Reverse so recall increases for AUC calculation
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])

    return precision, recall, thresholds


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # Sort by recall increasing
    order = np.argsort(recall)
    return trapezoidal_auc(recall[order], precision[order])


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.001) -> Tuple[float, float]:
    """Compute the maximum Recall achieved where False Positive Rate <= target_fpr.

    Returns:
        (recall, threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid_idxs = np.where(fpr <= target_fpr)[0]
    if len(valid_idxs) == 0:
        return 0.0, 1.0
    best_idx = valid_idxs[-1]
    return float(tpr[best_idx]), float(thresholds[best_idx])


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """Compute binary cross-entropy loss with numerical stabilization."""
    y_t = _to_numpy(y_true)
    y_p = np.clip(_to_numpy(y_prob), eps, 1.0 - eps)
    loss = -(y_t * np.log(y_p) + (1.0 - y_t) * np.log(1.0 - y_p))
    return float(np.mean(loss))


def brier_score_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute mean squared probability calibration error (Brier Score)."""
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_prob)
    return float(np.mean((y_p - y_t) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across probability bins."""
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_t)
    if total == 0:
        return 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        in_bin = (y_p >= low) & (y_p < high if i < n_bins - 1 else y_p <= high)
        n_in_bin = np.sum(in_bin)
        if n_in_bin > 0:
            avg_confidence = np.mean(y_p[in_bin])
            avg_accuracy = np.mean(y_t[in_bin])
            ece += (n_in_bin / total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Generate summary classification metrics dictionary."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "false_positive_rate": false_positive_rate(y_true, y_pred),
    }
