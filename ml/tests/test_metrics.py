"""Unit tests for from-scratch evaluation metrics."""

import unittest
import numpy as np

from ml.metrics import (
    accuracy_score,
    confusion_matrix,
    expected_calibration_error,
    f1_score,
    false_positive_rate,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_at_fpr,
    recall_score,
    roc_auc_score,
    roc_curve,
    trapezoidal_auc,
)


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([0, 0, 0, 1, 1, 1])
        self.y_pred = np.array([0, 0, 1, 0, 1, 1])
        self.scores = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])

    def test_confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        self.assertEqual(cm[0, 0], 2)  # TN
        self.assertEqual(cm[0, 1], 1)  # FP
        self.assertEqual(cm[1, 0], 1)  # FN
        self.assertEqual(cm[1, 1], 2)  # TP

    def test_precision_recall_f1(self):
        prec = precision_score(self.y_true, self.y_pred)
        rec = recall_score(self.y_true, self.y_pred)
        f1 = f1_score(self.y_true, self.y_pred)
        self.assertAlmostEqual(prec, 2 / 3)
        self.assertAlmostEqual(rec, 2 / 3)
        self.assertAlmostEqual(f1, 2 / 3)

    def test_trapezoidal_auc(self):
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.5, 1.0])
        auc = trapezoidal_auc(x, y)
        self.assertAlmostEqual(auc, 0.5)

    def test_roc_auc_score(self):
        auc = roc_auc_score(self.y_true, self.scores)
        self.assertGreaterEqual(auc, 0.5)
        self.assertLessEqual(auc, 1.0)

    def test_recall_at_fpr(self):
        rec, thresh = recall_at_fpr(self.y_true, self.scores, target_fpr=0.5)
        self.assertGreaterEqual(rec, 0.0)

    def test_expected_calibration_error(self):
        ece = expected_calibration_error(self.y_true, self.scores, n_bins=5)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)


if __name__ == "__main__":
    unittest.main()
