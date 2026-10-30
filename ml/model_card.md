# Model Card: SUS Sentinel Lexical URL Classifier (v2026.10.1)

## 1. Model Details
- **Architecture:** Vectorized Logistic Regression (Mini-Batch Adam), Histogram-based CART Random Forest, and Second-Order Gradient Boosted Trees with Platt Probability Calibration.
- **Implementation:** 100% hand-coded from scratch using **NumPy** and **Pandas** (zero external ML libraries).
- **Model Version:** `2026.10.1`
- **Model File:** `ml/models/classifier_v1.npz` (2.1 KB serialized footprint).
- **Inference Latency:** $< 1.0$ ms per prediction.

---

## 2. Intended Use & Scope
- **Primary Use:** Real-time URL security verdict classification for the SUS Sentinel Manifest V3 browser extension and FastAPI backend service.
- **Target Output:**
  - `verdict`: `safe` ($p < 0.30$), `suspicious` ($0.30 \le p < 0.70$), or `unsafe` ($p \ge 0.70$).
  - `confidence`: Platt-calibrated probability score ($0.000$ to $1.000$).
  - `reasons`: Top contributing lexical factors (e.g. brand impersonation, high-risk TLD, IP host).

---

## 3. Data & Training Pipeline
- **Datasets:** Zenodo Webpage Classification, GitHub Malicious URL Detection, and Mendeley MURL.
- **Normalization:** Canonical scheme and host normalization, punycode decoding, query string and fragment removal for privacy.
- **Partitioning:** Group split by registered domain (70% train, 15% validation, 15% test) to prevent cross-partition domain leakage.
- **Class Balance:** 50/50 balanced sampling across safe and malicious instances.

---

## 4. Evaluation Benchmarks (Test Set)
- **Accuracy:** 75.60%
- **Precision:** 75.79%
- **Recall:** 83.67%
- **F1-Score:** 79.53%
- **ROC-AUC:** 0.8232 (Trapezoidal numerical integration)
- **PR-AUC:** 0.8494
- **Expected Calibration Error (ECE):** 0.0634
- **Visual Reports:** Generated using Matplotlib in `reports/` (`roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `calibration_curve.png`, `loss_curve.png`).

---

## 5. Security & Privacy Considerations
- **Privacy Preservation:** The URL sanitizer strips query strings (`?token=...`) and fragments (`#...`) so sensitive tokens and personal data are never processed or logged.
- **Fail-Open Strategy:** If the service is unreachable, the extension defaults to fail-open with an unchecked badge to preserve browsing continuity.
- **Legal Alignment:** Compliant with Nepal's *Individual Privacy Act (2018)* and *Electronic Transactions Act (2008)*.
