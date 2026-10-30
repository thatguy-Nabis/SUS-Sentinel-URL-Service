"""Visualization utilities using Matplotlib exclusively."""

from typing import List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless servers
import matplotlib.pyplot as plt
import numpy as np

from ml.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, pr_auc_score


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Receiver Operating Characteristic (ROC)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot ROC curve with trapezoidal AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot Precision-Recall curve with PR-AUC."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc = pr_auc_score(y_true, y_score)
    prevalence = np.mean(y_true == 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#10b981", lw=2, label=f"PR curve (AUC = {auc:.4f})")
    ax.axhline(prevalence, color="#9ca3af", lw=1.5, linestyle="--", label=f"Baseline ({prevalence:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot confusion matrix heatmap."""
    if class_names is None:
        class_names = ["Safe (0)", "Unsafe (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12, fontweight="bold"
            )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Probability Calibration Reliability Diagram",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot reliability diagram comparing predicted vs empirical probabilities."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    true_proportions = []
    mean_confidences = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        in_bin = (y_prob >= low) & (y_prob < high if i < n_bins - 1 else y_prob <= high)
        if np.sum(in_bin) > 0:
            mean_confidences.append(float(np.mean(y_prob[in_bin])))
            true_proportions.append(float(np.mean(y_true[in_bin])))
        else:
            mean_confidences.append(float(bin_centers[i]))
            true_proportions.append(float(bin_centers[i]))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", label="Perfect Calibration")
    ax.plot(mean_confidences, true_proportions, marker="o", color="#8b5cf6", lw=2, label="Model Confidence")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Empirical Fraction of Positives")
    ax.set_title(title)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    return fig


def plot_learning_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Loss Progression",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot loss curve over iterations or epochs."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_losses, label="Train Loss", color="#2563eb", lw=1.8)
    if val_losses:
        ax.plot(val_losses, label="Validation Loss", color="#ef4444", lw=1.8)
    ax.set_xlabel("Iteration / Epoch")
    ax.set_ylabel("Log Loss")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    return fig
