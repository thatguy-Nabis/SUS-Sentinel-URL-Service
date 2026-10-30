"""Unit tests for from-scratch algorithms in pure NumPy."""

import unittest
import numpy as np

from ml.algorithms.logistic_regression import CustomLogisticRegression, CustomStandardScaler
from ml.algorithms.tree import BinMapper, DecisionTreeClassifier
from ml.algorithms.random_forest import RandomForestClassifier
from ml.algorithms.boosting import GradientBoostingClassifier
from ml.algorithms.calibration import PlattCalibrator, IsotonicCalibrator
from ml.algorithms.ensemble import WeightedEnsemble
from ml.metrics import accuracy_score


class TestAlgorithms(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        # Generate synthetic linearly separable classification dataset
        n = 200
        X0 = rng.normal(loc=-2.0, scale=0.8, size=(n // 2, 4))
        X1 = rng.normal(loc=2.0, scale=0.8, size=(n // 2, 4))
        self.X = np.vstack([X0, X1])
        self.y = np.array([0] * (n // 2) + [1] * (n // 2))

    def test_standard_scaler(self):
        scaler = CustomStandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        np.testing.assert_allclose(np.mean(X_scaled, axis=0), 0.0, atol=1e-7)
        np.testing.assert_allclose(np.std(X_scaled, axis=0), 1.0, atol=1e-7)

    def test_logistic_regression(self):
        scaler = CustomStandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        lr = CustomLogisticRegression(learning_rate=0.05, max_epochs=50, batch_size=32)
        lr.fit(X_scaled, self.y)

        preds = lr.predict(X_scaled)
        acc = accuracy_score(self.y, preds)
        self.assertGreater(acc, 0.90)
        self.assertLess(lr.loss_history_[-1], lr.loss_history_[0])

    def test_decision_tree(self):
        tree = DecisionTreeClassifier(max_depth=4, min_samples_split=5)
        tree.fit(self.X, self.y)
        preds = tree.predict(self.X)
        acc = accuracy_score(self.y, preds)
        self.assertGreater(acc, 0.90)

    def test_random_forest(self):
        rf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
        rf.fit(self.X, self.y)
        preds = rf.predict(self.X)
        acc = accuracy_score(self.y, preds)
        self.assertGreater(acc, 0.90)
        if rf.oob_score_:
            self.assertGreater(rf.oob_score_, 0.80)

    def test_gradient_boosting_depthwise(self):
        gb = GradientBoostingClassifier(
            n_estimators=15,
            learning_rate=0.1,
            max_depth=3,
            growth_policy="depthwise",
            random_state=42,
        )
        gb.fit(self.X, self.y)
        preds = gb.predict(self.X)
        acc = accuracy_score(self.y, preds)
        self.assertGreater(acc, 0.90)

    def test_gradient_boosting_lossguide(self):
        gb = GradientBoostingClassifier(
            n_estimators=15,
            learning_rate=0.1,
            max_depth=4,
            max_leaves=8,
            growth_policy="lossguide",
            random_state=42,
        )
        gb.fit(self.X, self.y)
        preds = gb.predict(self.X)
        acc = accuracy_score(self.y, preds)
        self.assertGreater(acc, 0.90)

    def test_platt_calibrator(self):
        scores = np.linspace(-3.0, 3.0, 50)
        y = (scores > 0.0).astype(int)
        cal = PlattCalibrator().fit(scores, y)
        probs = cal.predict_proba(scores)
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        # Monotonicity check
        self.assertTrue(np.all(np.diff(probs) >= 0.0))

    def test_isotonic_calibrator(self):
        scores = np.linspace(-3.0, 3.0, 50)
        y = (scores > 0.0).astype(int)
        cal = IsotonicCalibrator().fit(scores, y)
        probs = cal.predict_proba(scores)
        self.assertTrue(np.all(np.diff(probs) >= -1e-7))

    def test_weighted_ensemble(self):
        scaler = CustomStandardScaler()
        X_s = scaler.fit_transform(self.X)

        lr = CustomLogisticRegression(max_epochs=20).fit(X_s, self.y)
        tree = DecisionTreeClassifier(max_depth=3).fit(self.X, self.y)

        ensemble = WeightedEnsemble(
            estimators=[("lr", lr), ("tree", tree)],
            weights=[0.5, 0.5],
        )
        probs = ensemble.predict_proba(X_s)
        self.assertEqual(len(probs), len(self.y))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))


if __name__ == "__main__":
    unittest.main()
