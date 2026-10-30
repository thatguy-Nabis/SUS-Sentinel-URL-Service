import argparse
import json
import os
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from ml.algorithms.boosting import GradientBoostingClassifier
from ml.algorithms.calibration import PlattCalibrator
from ml.algorithms.ensemble import WeightedEnsemble
from ml.algorithms.logistic_regression import CustomLogisticRegression, CustomStandardScaler
from ml.algorithms.random_forest import RandomForestClassifier
from ml.features.extractor import LexicalFeatureExtractor
from ml.metrics import (
    classification_report,
    confusion_matrix,
    expected_calibration_error,
    pr_auc_score,
    recall_at_fpr,
    roc_auc_score,
)
from ml.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_precision_recall_curve,
    plot_roc_curve,
)


def load_dataset_features(
    csv_path: str, extractor: LexicalFeatureExtractor, max_samples: int = None
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load URLs from CSV and extract feature matrix."""
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    urls = df["url"].astype(str).tolist()
    labels = df["label"].to_numpy(dtype=np.int64)

    print(f"Extracting lexical features for {len(urls)} URLs...")
    start_time = time.time()
    X = extractor.extract_batch(urls)
    elapsed = time.time() - start_time
    print(f"Extraction complete in {elapsed:.2f}s ({len(urls) / max(0.001, elapsed):.1f} URLs/sec).")

    return X, labels, urls


def train_and_evaluate(
    data_dir: str = "data/processed",
    output_dir: str = "ml/models",
    reports_dir: str = "reports",
    max_train_samples: int = None,
) -> None:
    """Run full training, calibration, and evaluation cycle."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    extractor = LexicalFeatureExtractor()

    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing {train_path}. Run ml/training/data_prep.py first.")

    X_train, y_train, _ = load_dataset_features(train_path, extractor, max_train_samples)
    X_val, y_val, _ = load_dataset_features(val_path, extractor, max_train_samples)
    X_test, y_test, test_urls = load_dataset_features(test_path, extractor, max_train_samples)

    print("\nFitting StandardScaler on training data...")
    scaler = CustomStandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 1. Train Logistic Regression
    print("\n--- Training Custom Logistic Regression (Mini-Batch Adam) ---")
    lr = CustomLogisticRegression(
        learning_rate=0.03,
        lambda_reg=0.005,
        max_epochs=40,
        batch_size=1024,
        class_weight="balanced",
    )
    t0 = time.time()
    lr.fit(X_train_scaled, y_train, X_val=X_val_scaled, y_val=y_val)
    t_lr = time.time() - t0
    lr_val_auc = roc_auc_score(y_val, lr.predict_proba(X_val_scaled))
    print(f"Logistic Regression trained in {t_lr:.2f}s. Validation ROC-AUC: {lr_val_auc:.4f}")

    # 2. Train Random Forest
    print("\n--- Training Custom Random Forest (Histogram CART) ---")
    rf = RandomForestClassifier(
        n_estimators=20,
        max_depth=7,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    t_rf = time.time() - t0
    rf_val_auc = roc_auc_score(y_val, rf.predict_proba(X_val))
    print(f"Random Forest trained in {t_rf:.2f}s. Validation ROC-AUC: {rf_val_auc:.4f} (OOB: {rf.oob_score_:.4f})")

    # 3. Train Gradient Boosting
    print("\n--- Training Custom Gradient Boosting (Level-wise & Leaf-wise) ---")
    gb = GradientBoostingClassifier(
        n_estimators=30,
        learning_rate=0.1,
        max_depth=5,
        growth_policy="depthwise",
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=8,
        random_state=42,
    )
    t0 = time.time()
    gb.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    t_gb = time.time() - t0
    gb_val_auc = roc_auc_score(y_val, gb.predict_proba(X_val))
    print(f"Gradient Boosting trained in {t_gb:.2f}s. Validation ROC-AUC: {gb_val_auc:.4f}")

    # 4. Fit Ensemble
    print("\n--- Building Weighted Ensemble ---")
    ensemble = WeightedEnsemble(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
    )
    # Custom evaluation on validation for the ensemble
    lr_val_probs = lr.predict_proba(X_val_scaled)
    rf_val_probs = rf.predict_proba(X_val)
    gb_val_probs = gb.predict_proba(X_val)
    val_probs_matrix = np.column_stack([lr_val_probs, rf_val_probs, gb_val_probs])

    # Simple validation grid search for blending weights
    weights = np.array([0.25, 0.35, 0.40])
    ensemble.weights_ = weights
    raw_val_blend = np.dot(val_probs_matrix, weights)
    raw_val_blend_clipped = np.clip(raw_val_blend, 1e-6, 1.0 - 1e-6)
    val_logits = np.log(raw_val_blend_clipped / (1.0 - raw_val_blend_clipped))

    # 5. Fit Platt Calibration
    print("Fitting Platt Scaling calibrator on validation split...")
    calibrator = PlattCalibrator().fit(val_logits, y_val)
    ensemble.calibrator = calibrator
    print(f"Platt Calibrator parameters: a = {calibrator.a_:.4f}, b = {calibrator.b_:.4f}")

    # 6. Evaluation on Held-Out Test Set
    print("\n==========================================")
    print("     TEST SET EVALUATION & BENCHMARKS     ")
    print("==========================================")
    lr_test_probs = lr.predict_proba(X_test_scaled)
    rf_test_probs = rf.predict_proba(X_test)
    gb_test_probs = gb.predict_proba(X_test)

    raw_test_blend = (
        weights[0] * lr_test_probs
        + weights[1] * rf_test_probs
        + weights[2] * gb_test_probs
    )
    raw_test_blend_clipped = np.clip(raw_test_blend, 1e-6, 1.0 - 1e-6)
    test_logits = np.log(raw_test_blend_clipped / (1.0 - raw_test_blend_clipped))
    calibrated_test_probs = calibrator.predict_proba(test_logits)

    test_preds = (calibrated_test_probs >= 0.5).astype(int)

    metrics = classification_report(y_test, test_preds)
    test_roc_auc = roc_auc_score(y_test, calibrated_test_probs)
    test_pr_auc = pr_auc_score(y_test, calibrated_test_probs)
    recall_at_01_fpr, thresh_01 = recall_at_fpr(y_test, calibrated_test_probs, target_fpr=0.001)
    ece = expected_calibration_error(y_test, calibrated_test_probs)

    print(f"Test Accuracy:          {metrics['accuracy'] * 100:.2f}%")
    print(f"Test Precision:         {metrics['precision'] * 100:.2f}%")
    print(f"Test Recall:            {metrics['recall'] * 100:.2f}%")
    print(f"Test F1-Score:          {metrics['f1'] * 100:.2f}%")
    print(f"Test False Pos. Rate:   {metrics['false_positive_rate'] * 100:.3f}%")
    print(f"Test ROC-AUC:           {test_roc_auc:.4f}")
    print(f"Test PR-AUC:            {test_pr_auc:.4f}")
    print(f"Recall @ FPR <= 0.1%:   {recall_at_01_fpr * 100:.2f}% (Threshold: {thresh_01:.4f})")
    print(f"Expected Calib. Error:  {ece:.4f}")

    cm = confusion_matrix(y_test, test_preds)
    print(f"\nConfusion Matrix:\nTN={cm[0, 0]} | FP={cm[0, 1]}\nFN={cm[1, 0]} | TP={cm[1, 1]}")

    # 7. Generate Visual Reports via Matplotlib
    print("\nGenerating Matplotlib evaluation reports in reports/ ...")
    plot_roc_curve(
        y_test,
        calibrated_test_probs,
        title=f"Test ROC Curve (AUC = {test_roc_auc:.4f})",
        save_path=os.path.join(reports_dir, "roc_curve.png"),
    )
    plot_precision_recall_curve(
        y_test,
        calibrated_test_probs,
        title=f"Test Precision-Recall Curve (PR-AUC = {test_pr_auc:.4f})",
        save_path=os.path.join(reports_dir, "pr_curve.png"),
    )
    plot_confusion_matrix(
        cm,
        class_names=["Benign (0)", "Malicious (1)"],
        title="Test Confusion Matrix",
        save_path=os.path.join(reports_dir, "confusion_matrix.png"),
    )
    plot_calibration_curve(
        y_test,
        calibrated_test_probs,
        title=f"Probability Calibration (ECE = {ece:.4f})",
        save_path=os.path.join(reports_dir, "calibration_curve.png"),
    )
    plot_learning_curves(
        lr.loss_history_,
        lr.val_loss_history_,
        title="Logistic Regression Loss Progression",
        save_path=os.path.join(reports_dir, "loss_curve.png"),
    )
    print("Reports successfully generated.")

    # 8. Save Artifacts
    print("\nSerializing production model artifacts to ml/models/ ...")
    artifact_path = os.path.join(output_dir, "classifier_v1.npz")
    metadata_path = os.path.join(output_dir, "metadata.json")

    np.savez_compressed(
        artifact_path,
        lr_weights=lr.weights_,
        lr_bias=np.array([lr.bias_], dtype=np.float64),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        platt_a=np.array([calibrator.a_], dtype=np.float64),
        platt_b=np.array([calibrator.b_], dtype=np.float64),
        ensemble_weights=weights,
    )

    metadata = {
        "model_version": "2026.10.1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "features": extractor.feature_names,
        "metrics": {
            "test_accuracy": float(metrics["accuracy"]),
            "test_precision": float(metrics["precision"]),
            "test_recall": float(metrics["recall"]),
            "test_f1": float(metrics["f1"]),
            "test_roc_auc": float(test_roc_auc),
            "test_pr_auc": float(test_pr_auc),
            "recall_at_01_fpr": float(recall_at_01_fpr),
            "expected_calibration_error": float(ece),
        },
        "thresholds": {
            "safe_upper": 0.30,
            "suspicious_upper": 0.70,
            "strict_recall_threshold": float(thresh_01),
        },
        "calibrator": calibrator.to_dict(),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {artifact_path}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate from-scratch URL classifier.")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="ml/models")
    parser.add_argument("--reports_dir", default="reports")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    train_and_evaluate(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        max_train_samples=args.max_samples,
    )
