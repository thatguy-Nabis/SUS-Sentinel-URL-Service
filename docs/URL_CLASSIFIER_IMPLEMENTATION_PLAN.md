# URL Classifier: Phase-Wise Implementation Plan and Gap Analysis

Based on `CONTEXT.md`. Timelines assume a small team (1-3 people) working part-time, about 15 weeks end to end. Adjust to your real deadline.

**Project constraint (applies to the whole plan):** all classification algorithms are **hand-coded from scratch**. The only third-party libraries allowed for the ML and data work are **NumPy, pandas, and matplotlib** (matplotlib for visualization only). Python's standard library is assumed available. No scikit-learn, XGBoost, LightGBM, SciPy, imbalanced-learn, joblib, or similar. See section 2 for what this changes.

---

## 1. Summary

**What is being built:** a browser extension that classifies a URL as safe or unsafe (optionally with a category such as Phishing, Malware, Spam, Adult, Benign) and a confidence score. The classifier is an ensemble of **our own implementations** of Logistic Regression, Random Forest, and two gradient-boosted tree models (an XGBoost-style and a LightGBM-style variant). A small website supports the extension (Home, How It Works, Privacy Policy, About/Contact).

**Recommended MVP scope:**

1. Lexical-feature pipeline written with NumPy and pandas.
2. From-scratch models: Logistic Regression, CART decision tree, Random Forest, histogram-based gradient boosting (with a level-wise "XGBoost-style" and a leaf-wise "LightGBM-style" growth mode), plus a simple ensemble.
3. From-scratch calibration (Platt scaling, optionally isotonic) and evaluation metrics, with matplotlib plots.
4. Backend that loads the saved model and serves predictions.
5. Manifest V3 extension with a badge, warning page, consent screen, and "report wrong verdict" button.
6. Static website with the four required pages.

**Defer to later:** content-based features, live WHOIS lookups, multi-class categories beyond what the data supports, serverless or Kubernetes hosting, demo page.

---

## 2. What the "from scratch" constraint changes

| Area                    | Typical approach                              | Approach under this constraint                                                                                                                                                                                                                                                         |
| ----------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Models                  | scikit-learn, XGBoost, LightGBM               | Own implementations in NumPy. The XGBoost and LightGBM models become "XGBoost-style" and "LightGBM-style" gradient boosting: same ideas (second-order gradient boosting, histogram binning, level-wise vs leaf-wise growth), not the real libraries. Name them accordingly in reports. |
| Model files             | joblib, pickle, ONNX                          | `np.savez` for arrays plus a JSON file for structure and metadata. This also makes the model easy to port to JavaScript (see Phase 5).                                                                                                                                                 |
| Data files              | Parquet                                       | **CSV or `.npz`**. Parquet needs `pyarrow`, which is an extra library.                                                                                                                                                                                                                 |
| Splitting, CV, SMOTE    | sklearn, imbalanced-learn                     | Hand-written group and time splits with pandas and NumPy. **Drop SMOTE**: a hand-written k-nearest-neighbor SMOTE is slow and of questionable value. Use class weights and undersampling.                                                                                              |
| Calibration and metrics | sklearn                                       | Hand-written Platt scaling, isotonic regression (pool-adjacent-violators), ROC and PR curves, AUC, F1, confusion matrix, calibration error. Plots via matplotlib.                                                                                                                      |
| Testing                 | pytest                                        | Python's built-in `unittest`.                                                                                                                                                                                                                                                          |
| Tuning                  | sklearn search utilities                      | Own random or grid search loop on a subsample.                                                                                                                                                                                                                                         |
| WHOIS and DNS           | python-whois and similar                      | Standard library `socket` for raw lookups, or skip. Host features are Tier B and deferred anyway.                                                                                                                                                                                      |
| Backend framework       | FastAPI                                       | **Decision needed (see D1 in section 3).**                                                                                                                                                                                                                                             |
| Training time           | The estimates in `CONTEXT.md` (30-60 min CPU) | **No longer valid.** Pure NumPy tree building is considerably slower than the C++ libraries. Benchmark on a 100K subsample in Phase 3 and extrapolate. Expect tuning to run on subsamples and the final fit on the full set.                                                           |

---

## 3. Gap Analysis

Gaps are things `CONTEXT.md` leaves out, leaves ambiguous, or gets wrong. They are sorted by how much they can hurt the project. **Critical** items should be settled in Phase 0.

### 3.1 Critical

| #   | Gap                                                                                                                                                                                                        | Why it matters                                                                                                                                                                                                                                           | Recommended resolution                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C1  | **Inference-time feature availability.** Host-based (WHOIS, domain age, IP reputation) and content-based (HTML, JS) features are listed for training, but the extension only has a URL at prediction time. | Fetching WHOIS or page content per request adds seconds of latency, hits rate limits, and for content means the backend downloads possibly malicious pages (SSRF and safety risk). Training on features you cannot compute live causes train/serve skew. | Use a **tiered feature design**: Tier A lexical only (always available, MVP). Tier B host features from cached or batch-enriched sources (optional). Tier C content features offline research only. Train and serve the same feature code.                                                                                                                                                                                                                                          |
| C2  | **Privacy design of the core flow.** Every browsed URL is sent to your server, which is effectively a browsing history.                                                                                    | Conflicts with the stated rules (consent, minimal data) and with the Individual Privacy Act 2018. It is also the first thing store reviewers and users will question.                                                                                    | Strip query strings and fragments before sending. Cache locally and check a local allowlist first. **With hand-coded models, an on-device first pass is practical** because Logistic Regression and trees are just arrays and thresholds (see Phase 5). Require explicit opt-in consent. Do not log full URLs by default.                                                                                                                                                           |
| C3  | **Dataset bias and leakage.** Datasets come from different sources (benign from top-site lists, malicious from phishing feeds), and duplicates across the three are likely.                                | A model can learn source artifacts (for example `https` presence, path depth, TLD mix) instead of maliciousness, scoring 99% offline and failing on real traffic. Random train/test splits leak the same domains into both sets.                         | Deduplicate by normalized URL, then **split by domain and by time**. Check for source artifacts. Add a hold-out of "real-world-like" benign URLs (long paths, query strings, shorteners, login pages).                                                                                                                                                                                                                                                                              |
| C4  | **No success metrics.** The document gives training times but no accuracy, precision, recall, or latency targets.                                                                                          | You cannot decide when the model is "done" or pick a threshold. In real browsing, malicious URLs are well under 1% of traffic, so even a 1% false positive rate floods users with warnings.                                                              | Define targets up front, for example: false positive rate under 0.1% on popular-site traffic, recall at that FPR, p95 latency under 300 ms, per-class F1. Evaluate on the out-of-time hold-out.                                                                                                                                                                                                                                                                                     |
| C5  | **Hand-coded algorithms: correctness and speed risk.** There are no reference libraries to trust, and four model families is a lot to write in NumPy.                                                      | Bugs in a from-scratch tree or booster silently produce a weaker model rather than an error. Naive Python loops over 2M rows will make training impractically slow.                                                                                      | Write algorithms **vectorized with NumPy**, use **histogram binning** for trees (pre-bin each feature into about 64-256 bins, then use `np.bincount`), train on subsamples while developing, and validate each algorithm with unit tests (toy datasets with known answers, gradient checks for Logistic Regression, loss decreasing monotonically for boosting, tree predictions matching a hand-computed example). Share one tree builder between Random Forest and both boosters. |

### 3.2 High

| #   | Gap                                                                                                                                                                                             | Why it matters                                                                                                        | Recommended resolution                                                                                                                                                                                                                 |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| H1  | **Dataset size claim is inconsistent.** 1.07M + 649K + 783K is about 2.5M **before** cleaning, not after.                                                                                       | After deduplication and label cleaning the real count will be smaller, which affects training time and class balance. | Report the real post-cleaning counts per class in Phase 1.                                                                                                                                                                             |
| H2  | **Label taxonomy mismatch.** Datasets use different labels (Benign vs Legitimate, Defacement, Exploit, Adult, Spam, 20 sublabels). "Adult" is a content policy category, not a security threat. | Multi-class training on inconsistent labels gives noisy classes and misleading confidence.                            | Build an explicit **label mapping table**. Ship binary (safe/unsafe) first, then add categories only where class counts are adequate. Decide whether "Adult" counts as unsafe or is a separate optional filter.                        |
| H3  | **Confidence score is not defined.** Raw ensemble outputs are not probabilities. The `0.84` in the example result has no stated meaning.                                                        | Users and thresholds will treat 0.84 as an 84% chance, which is false for uncalibrated models.                        | Hand-code **Platt scaling** (one-feature logistic regression on held-out scores) and optionally **isotonic regression** (pool-adjacent-violators). Plot reliability curves and report expected calibration error.                      |
| H4  | **No complement to ML.** ML alone misses new campaigns and over-flags legitimate sites.                                                                                                         | Industry tools combine models with reputation lists.                                                                  | Add a **top-domain allowlist** (for example Tranco top N) and free blocklist feeds (PhishTank, URLhaus, OpenPhish, subject to their terms). ML handles what lists do not know.                                                         |
| H5  | **Extension behavior is under-specified.** "Displays badge" is the entire description.                                                                                                          | Missing: when it checks, what blocking looks like, failure behavior, user override, and false positive reporting.     | Define: check on navigation start, warning interstitial for unsafe, badge for all, **fail-open with an "unchecked" badge** if the backend is down (fail-closed as a user setting), allow override, and include "Report wrong verdict". |
| H6  | **Compliance items are listed but not turned into deliverables.**                                                                                                                               | "Obtain consent" and "secure logs" need concrete screens, policies, and settings.                                     | See Phase 7. Have the legal text reviewed by someone qualified. This document is not legal advice.                                                                                                                                     |
| H7  | **Dataset licensing.** The three sources have their own licenses and terms.                                                                                                                     | Copyright is in your compliance list, but redistribution, commercial use, and attribution rules are not checked.      | Record each dataset's license and required attribution in Phase 1. Do not redistribute raw data in the repo or on the website.                                                                                                         |

### 3.3 Medium

| #   | Gap                                                                                                                                                                         | Resolution                                                                                                                                                                                                                                                         |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| M1  | **Evasion cases not covered**: URL shorteners, redirect chains, IDN/homoglyph domains, legitimate hosting abused for phishing (github.io, Google Sites), data/blob URLs.    | Add a URL normalizer (standard library `urllib.parse`), punycode handling, and optional shortener expansion (server-side, with SSRF protections). Track these as a test set.                                                                                       |
| M2  | **Ensemble design is vague, and the work multiplies under the constraint.** LR, RF, and two boosters overlap heavily, and a majority vote of correlated models adds little. | Build the shared tree builder once. Start with each model as a baseline, then compare weighted average vs a stacked logistic regression. Keep the ensemble only if it beats the best single model on the hold-out. If time runs short, ship LR + RF + one booster. |
| M3  | **Class imbalance handling.** SMOTE is dropped (see section 2).                                                                                                             | Use class weights in the loss and split criteria, plus undersampling of the majority class. Never alter validation or test data.                                                                                                                                   |
| M4  | **Model size and speed.** Large forests are big and slow to load and score. Python-level tree traversal is slow.                                                            | Cap tree count and depth. Store trees as flat NumPy arrays (feature, threshold, left, right, value) and **predict in a vectorized way** (advance all rows one level at a time). Prefer a small always-on container over serverless.                                |
| M5  | **API access control.** An extension cannot hold a secret. Anyone can call your public API.                                                                                 | Rate limit per IP and per install ID, validate input size, add abuse monitoring.                                                                                                                                                                                   |
| M6  | **Model lifecycle missing**: versioning, monitoring, drift, retraining, rollback.                                                                                           | Version models (`model_version` in every response), log aggregate metrics (not URLs), keep a retraining schedule and a feedback queue.                                                                                                                             |
| M7  | **Hosting and latency for Nepal users.**                                                                                                                                    | Choose a region close to users (a Mumbai-region cloud is usually closest), measure round-trip from Nepal, cache aggressively.                                                                                                                                      |
| M8  | **Website content vs the earlier Stitch design.** The design includes stats, team cards, and FAQ content that are placeholders.                                             | Replace placeholder numbers with real measured metrics from Phase 3 or remove them. Add Terms of Use alongside the Privacy Policy.                                                                                                                                 |
| M9  | **Cross-browser support.**                                                                                                                                                  | Build for Chrome (Manifest V3) first. Edge uses the same package. Firefox needs manifest and API adjustments.                                                                                                                                                      |
| M10 | **Testing is not mentioned.**                                                                                                                                               | Use `unittest` for feature, algorithm, and API tests, plus extension end-to-end checks, a load test, and a regression set of known URLs.                                                                                                                           |
| M11 | **Training reproducibility.** Hand-written algorithms with random sampling are hard to debug if runs differ.                                                                | Seed `np.random.default_rng` everywhere, log hyperparameters and metrics per run to a CSV, and save plots with matplotlib.                                                                                                                                         |
| M12 | **Feature parity if the extension runs the model locally.** The feature extractor would exist in Python and JavaScript.                                                     | Create a shared file of golden test cases (URL and expected feature vector) and run it against both implementations.                                                                                                                                               |

### 3.4 Decisions and open questions to settle before building

1. **D1: Does the "numpy, pandas, matplotlib only" rule cover the backend web framework?** Recommended reading: the rule applies to the ML and data code, and the API layer is just transport, so FastAPI with uvicorn is acceptable. If your supervisor or rubric means _every_ dependency, serve with the standard library `http.server` instead (slower and less polished but fully compliant). Pick one in Phase 0.
2. **D2: Is it acceptable to cross-check your implementations against scikit-learn in a throwaway scratch environment** (never in the project code or requirements)? It is a very effective way to catch bugs, but only if the rule allows it. If not, rely on unit tests with hand-computed examples.
3. **D3: Is this the same product as Sentinelkey** (Express/TypeScript API, Flask ML service, MV3 extension, dashboard), or a separate project? If it is the same, the from-scratch model becomes the ML service in that stack and should use one consistent name.
4. **D4: Binary or multi-class at launch?** (Recommendation: binary first.)
5. **D5: Who is the audience and deadline?** (Academic submission, public release, or both.) This decides how much of Phases 8 and 9 you do.
6. **D6: Will you publish to the Chrome Web Store?** If yes, plan for its privacy disclosures and review time.
7. **D7: What is the policy on user URL logging?** (None, aggregate only, or opt-in samples for retraining.)

---

## 4. Phase Plan

### Phase 0: Scoping and decisions (Week 1)

**Goal:** lock the decisions that are expensive to change later.

**Tasks**

- Answer D1 to D7 in section 3.4.
- Choose MVP scope (section 1) and success metrics (C4).
- Choose the privacy model (C2) and the feature tiers (C1).
- Define the API contract (section 6) and the label taxonomy (H2).
- Set up the repo, `requirements.txt` (numpy, pandas, matplotlib only, plus the backend decision from D1), and folder structure (section 7).

**Deliverables:** decision log, metric targets, API contract draft, repo skeleton.

**Exit criteria:** every Critical gap has a written decision.

---

### Phase 1: Data acquisition and preparation (Weeks 2-3)

**Goal:** a clean, deduplicated, correctly split dataset.

**Tasks**

- Download the three datasets and record license, source, date, and column schema for each. Confirm which contain only URLs and which include host or HTML data.
- Normalize URLs with `urllib.parse` and pandas string operations (lowercase host, strip fragments, handle punycode, remove trailing noise) and deduplicate across all datasets.
- Apply the label mapping table (H2) and drop ambiguous labels.
- Add a realistic benign set (for example Tranco top domains with sampled real paths) so the benign class is not only homepages.
- Write the **split code by hand**: group by registered domain, then assign whole groups to train, validation, and test; keep a separate out-of-time hold-out where dates exist.
- Exploratory analysis with pandas and matplotlib: class balance per source, URL length distributions, per-source artifact checks (C3).
- Save processed data as **CSV** (or `.npz` for numeric matrices), not Parquet.

**Deliverables:** `data/processed/` CSV files, `data_card.md` (sources, licenses, counts, mapping, split method), EDA notebook or script with saved plots.

**Exit criteria:** post-cleaning counts reported (H1), no domain appears in more than one split, licenses recorded.

---

### Phase 2: Feature engineering (Weeks 3-4)

**Goal:** one shared feature pipeline used in both training and serving, written with NumPy, pandas, and the standard library.

**Tasks**

- **Tier A (lexical, MVP):** URL length, host length, path length, number of dots, hyphens, digits, special characters, subdomain count, TLD, IP-as-host flag, `@` symbol, character entropy, suspicious keyword hits (login, secure, update, verify), brand lookalike distance (hand-coded edit distance), `https` flag, shortener flag. Optional: hashed character n-gram counts for Logistic Regression only.
- Encode categorical features (such as TLD) by hand: frequency or target-rate encoding computed on the training split only, to avoid leakage.
- **Tier B (host, optional):** domain age and registrar via cached or batch enrichment. WHOIS data is often redacted, so add a "missing" indicator.
- **Tier C (content, research only):** HTML tag counts, JavaScript presence, form and iframe counts. Not enabled in production without the safety review in C1.
- Package the extractor as a single Python module, importable by both training and API code, with `unittest` cases on fixed example URLs.
- Produce the golden test cases file for later JavaScript parity (M12).
- Pre-compute **histogram bin edges** per feature (quantile bins) on the training set for the tree models and save them with the model.

**Deliverables:** `features/` module, feature dictionary, tests, golden cases file, saved bin edges.

**Exit criteria:** identical feature output from the training pipeline and the API code path.

---

### Phase 3: From-scratch model implementation, training, and evaluation (Weeks 4-9)

**Goal:** a calibrated ensemble, built with only NumPy, pandas, and matplotlib, that meets the Phase 0 metrics.

#### 3A. Core utilities (Week 4-5)

- Vectorized sigmoid (numerically stable), log-loss, weighted loss helpers.
- Metric functions: confusion matrix, precision, recall, F1, ROC curve and AUC, precision-recall curve and AUC, recall at a fixed false positive rate, expected calibration error.
- Plot helpers (matplotlib): ROC, PR, calibration curve, feature importance, learning curves, confusion matrix.
- Seeded random number handling and a run logger (hyperparameters and metrics to CSV).

#### 3B. Logistic Regression (Week 5)

- Mini-batch gradient descent with L2 regularization, class weights, feature standardization, and learning-rate decay (optionally Adam, written by hand).
- Verify with a numerical gradient check and a toy linearly separable dataset.
- Export weights, bias, and scaler parameters (these are what the on-device extension would load).

#### 3C. Decision tree and Random Forest (Weeks 5-6)

- Histogram-based CART: pre-binned features, per-node class histograms via `np.bincount`, Gini or entropy split search over all bins vectorized, max depth, min samples per leaf, class weights.
- Store trees as flat arrays (feature, threshold bin, left child, right child, leaf value) and write **vectorized prediction**.
- Random Forest: bootstrap sampling, per-split feature subsampling, probability averaging, out-of-bag score as a free validation signal.
- Verify against a hand-computed tiny example and check that deeper trees fit the training data better.

#### 3D. Gradient boosting: "XGBoost-style" and "LightGBM-style" (Weeks 6-8)

- One histogram-based boosting engine using **gradients and hessians** of log-loss, with L2 leaf regularization, shrinkage (learning rate), and row and column subsampling.
- Growth policy switch: **level-wise** (XGBoost-style) and **leaf-wise, best-first** with a max-leaves limit (LightGBM-style).
- Early stopping on the validation set.
- Verify that training loss falls steadily each round and that a single boosting round on a toy dataset matches a hand-calculated leaf value.
- Benchmark on a 100K subsample early, then project full-data training time. Replace the `CONTEXT.md` estimates with measured ones.

#### 3E. Ensemble, calibration, and evaluation (Weeks 8-9)

- Compare ensemble strategies: simple average, weighted average (weights tuned on validation), and stacking with a hand-written logistic regression meta-model. Keep it only if it beats the best single model (M2).
- Calibrate with Platt scaling and optionally isotonic regression (H3). Plot reliability curves before and after.
- Choose decision thresholds to hit the target false positive rate on popular-site traffic (C4).
- Evaluate on the test set and the out-of-time hold-out. Report PR-AUC, recall at the FPR target, per-class metrics, confusion matrix, and error analysis (which kinds of URLs fail). Include evasion cases (M1).
- Generate **reasons** for the API response: for Logistic Regression, coefficient times feature value; for trees, split-gain importance or a simple path contribution method written by hand.
- Save the final model as `.npz` arrays plus a JSON metadata file (feature names, bin edges, scaler, thresholds, calibration parameters, `model_version`). Write the model card.

**Deliverables:** the `ml/` package with algorithms, unit tests, trained model files, evaluation report with matplotlib figures, `model_card.md`, run log CSV.

**Exit criteria:** metrics meet targets on the hold-out, calibration verified, algorithm unit tests pass, model size and inference time within budget.

---

### Phase 4: Backend service (Weeks 8-10)

**Goal:** a secure, observable classification API that loads the saved model.

**Tasks**

- Build the app (framework per D1) with `POST /v1/classify`, `GET /v1/health`, and `POST /v1/report`.
- Request pipeline: validate and sanitize URL, normalize, allowlist check, blocklist check, cache lookup, feature extraction, vectorized model inference, calibrated score, threshold to verdict, response with reasons.
- Load the model once at startup from `.npz` and JSON. No pickle, to avoid unsafe deserialization.
- Add a cache (in-process dictionary with expiry, or Redis if allowed), rate limiting, input size limits, CORS configuration for the extension, and HTTPS only.
- Logging policy: log request metadata and verdicts, **not raw URLs** by default. Store hashed URLs if you need deduplication. Set a retention period.
- Dockerize with a health check. Write `unittest` tests for the API, plus a load test script.
- Add basic metrics (request count, latency, verdict distribution, cache hit rate).

**Deliverables:** running container, API docs, test suite, load test results.

**Exit criteria:** p95 latency within target under load, rate limiting works, no raw URLs in logs.

---

### Phase 5: Browser extension (Weeks 9-12)

**Goal:** a usable, privacy-respecting Manifest V3 extension.

**Tasks**

- Manifest V3 setup with minimal permissions. Justify each permission in writing for store review.
- Background service worker: capture navigation, sanitize the URL (strip query and fragment), check local cache and allowlist, call the API, cache the result.
- UI: toolbar badge (green safe, red unsafe, gray unchecked), popup with verdict, confidence, and reasons, and a warning interstitial for unsafe URLs with "Go back" and "Proceed anyway".
- First-run **consent screen** explaining what is sent and why, with an opt-out. Settings page for fail-open vs fail-closed, strictness level, and clearing the cache.
- "Report wrong verdict" button wired to `/v1/report`.
- Handle backend downtime, timeouts, and offline mode gracefully.
- **Optional on-device mode (strongly recommended):** because the models are plain arrays, port the feature extractor and the Logistic Regression (and optionally tree) inference to JavaScript. Load weights from the JSON file. Most URLs are then classified locally and never leave the browser, which addresses C2 directly. Validate with the golden test cases (M12).
- Test on Chrome and Edge, then evaluate Firefox.

**Deliverables:** packaged extension, permission justification doc, test checklist, store screenshots.

**Exit criteria:** end-to-end flow works on real browsing, the extension never blocks the browser when the backend is down (unless fail-closed is selected), the consent screen cannot be skipped, and on-device and server predictions agree on the golden cases.

---

### Phase 6: Website (Weeks 11-13, overlaps Phase 5)

**Goal:** a trustworthy support site using the Stitch designs.

**Tasks**

- Pages: Home, How It Works, Privacy Policy, About, Contact (and Terms of Use).
- Make How It Works match the real pipeline, including limitations and what is and is not collected. Mention that the models are implemented from scratch if that is a selling point or an academic requirement.
- Replace placeholder stats and team content with real information, including real metrics and matplotlib charts from Phase 3 where useful.
- Contact form backend (or a mailto and a form service), with spam protection.
- Optional: demo page (paste URL, get verdict) that calls the same API, with rate limiting and a clear "not a guarantee" disclaimer.
- Host as a static site with HTTPS. Add the extension download link once the store listing exists.

**Deliverables:** deployed website, content review checklist.

**Exit criteria:** every claim on the site is true and matches the product's behavior.

---

### Phase 7: Compliance and security hardening (Weeks 13-14, started earlier)

**Goal:** turn the compliance list into verifiable controls.

**Checklist**

- [ ] Consent captured before any URL is sent (extension first run and website policy).
- [ ] Data minimization documented: what is sent (sanitized URL, client version), what is never sent (page content, cookies, full history).
- [ ] Privacy Policy matches actual behavior and covers: data collected, purpose, retention, third parties, user rights, contact. Reviewed by a qualified person against the Individual Privacy Act 2018.
- [ ] Electronic Transactions Act 2008 and National Cyber Security Policy 2023 reviewed for obligations on the service and for secure operation.
- [ ] Reporting process for confirmed malicious URLs aligned with Cyber Bureau guidance (confirm the current process and contact).
- [ ] Dataset licenses and attributions satisfied (H7). Copyright Act 2002 considerations for any scraped or third-party content.
- [ ] Liability wording: classifications are probabilistic and may be wrong. Include a dispute and correction path for site owners.
- [ ] API security: HTTPS, rate limits, input validation, dependency review, secrets management, log access control.
- [ ] Threat model written (API abuse, poisoning via the report endpoint, SSRF if any fetching exists, model file integrity).
- [ ] Report endpoint protected against data poisoning (reports reviewed before entering training data).

**Deliverables:** compliance checklist signed off, threat model, final privacy policy and terms.

---

### Phase 8: Deployment and release (Weeks 14-15)

**Goal:** a stable public release.

**Tasks**

- Deploy the backend container (a single small always-on instance, or Render, Fly, or a small cloud VM). Add HTTPS, a domain, and monitoring and alerts.
- CI/CD: build, run `unittest`, and deploy. Pin the model version.
- Submit the extension to the Chrome Web Store (and Edge Add-ons) with the privacy disclosure and permission justifications. Allow review time.
- Prepare a rollback plan for both the model and the extension.
- Smoke test the production path from a Nepal network connection.

**Deliverables:** live API, live website, store submission, runbook.

**Exit criteria:** end-to-end production flow verified, alerts firing correctly, rollback tested.

---

### Phase 9: Post-launch operations (ongoing)

- Review aggregate verdict distribution and false positive reports weekly.
- Retrain on a schedule using new feeds and **reviewed** reports. Compare the new model with the current one on the fixed hold-out before promotion.
- Monitor drift (feature distribution shift, rising false positive reports).
- Update the website metrics and privacy policy when behavior changes.
- Roadmap candidates: multi-class categories, Tier B host features, Firefox, user allowlist sync, faster training (more aggressive binning, feature selection).

---

## 5. Timeline Overview

| Phase                                    | Weeks   | Depends on                             |
| ---------------------------------------- | ------- | -------------------------------------- |
| 0. Scoping and decisions                 | 1       | None                                   |
| 1. Data preparation                      | 2-3     | 0                                      |
| 2. Feature engineering                   | 3-4     | 1 (partly parallel)                    |
| 3A. Core utilities and metrics           | 4-5     | 0                                      |
| 3B. Logistic Regression                  | 5       | 2, 3A                                  |
| 3C. Decision tree and Random Forest      | 5-6     | 2, 3A                                  |
| 3D. Gradient boosting (two growth modes) | 6-8     | 3C (shared tree builder)               |
| 3E. Ensemble, calibration, evaluation    | 8-9     | 3B-3D                                  |
| 4. Backend service                       | 8-10    | API contract from 0 (stub model first) |
| 5. Browser extension                     | 9-12    | API contract, stub backend             |
| 6. Website                               | 11-13   | Design, real metrics from 3E           |
| 7. Compliance and hardening              | 13-14   | 4, 5, 6 (start policy drafts earlier)  |
| 8. Deployment and release                | 14-15   | All                                    |
| 9. Post-launch                           | Ongoing | 8                                      |

**Parallelism tip:** once the API contract is fixed in Phase 0, the extension and website can be built against a stub backend that returns canned verdicts while the models are still being written.

**Schedule risk:** Phase 3 is the largest and least predictable block because every algorithm is written by hand. If time runs short, cut in this order: LightGBM-style growth mode, stacking, isotonic calibration, then the second booster.

---

## 6. Proposed API Contract (v1)

Extends the example in `CONTEXT.md`.

**Request:** `POST /v1/classify`

```json
{
  "url": "http://login-update-security.com/verify",
  "client_version": "0.1.0",
  "mode": "binary"
}
```

The extension should strip query strings and fragments before sending.

**Response:**

```json
{
  "verdict": "unsafe",
  "category": "phishing",
  "confidence": 0.84,
  "model_version": "2026.10.1",
  "source": "model",
  "reasons": [
    "Suspicious keywords in domain",
    "Hyphenated multi-word domain",
    "No HTTPS"
  ],
  "cached": false
}
```

Notes:

- `verdict` is one of `safe`, `suspicious`, `unsafe`, `unknown`.
- `category` is `null` in binary mode.
- `source` is one of `allowlist`, `blocklist`, `model`, `on_device`.
- `confidence` is a calibrated probability, not a raw model score (H3).
- `reasons` come from the feature-contribution method in Phase 3E.

---

## 7. Suggested Repository Structure

```
url-classifier/
├── data/
│   ├── raw/                  # not committed; documented in data_card.md
│   └── processed/            # CSV / .npz
├── notebooks/                # EDA and experiments (pandas + matplotlib)
├── ml/
│   ├── features/             # shared feature extractor (train + serve)
│   ├── algorithms/
│   │   ├── logistic_regression.py
│   │   ├── tree.py           # histogram CART, shared builder
│   │   ├── random_forest.py
│   │   ├── boosting.py       # gradient boosting, level-wise + leaf-wise
│   │   ├── ensemble.py
│   │   └── calibration.py    # Platt scaling, isotonic (PAV)
│   ├── metrics.py            # ROC, PR, AUC, F1, calibration error
│   ├── plots.py              # matplotlib figures
│   ├── training/             # train, tune, evaluate scripts
│   ├── tests/                # unittest cases for every algorithm
│   ├── models/               # versioned .npz + JSON artifacts
│   └── model_card.md
├── backend/
│   ├── app/
│   ├── tests/
│   └── Dockerfile
├── extension/
│   ├── manifest.json
│   ├── background/
│   ├── popup/
│   ├── warning/
│   ├── options/
│   └── ondevice/             # JS feature extractor + model inference
├── shared/
│   └── golden_cases.json     # URL -> expected features, used by Python and JS tests
├── website/                  # static pages from the Stitch designs
├── docs/
│   ├── decisions.md
│   ├── threat_model.md
│   ├── compliance_checklist.md
│   └── data_card.md
└── .github/workflows/        # CI/CD
```

---

## 8. Risk Register

| Risk                                                           | Likelihood | Impact | Mitigation                                                                                                            |
| -------------------------------------------------------------- | ---------- | ------ | --------------------------------------------------------------------------------------------------------------------- |
| Bugs in hand-written algorithms silently weaken the model (C5) | High       | High   | Unit tests with hand-computed examples, gradient checks, loss monotonicity checks, optional scratch cross-check (D2)  |
| Pure NumPy training too slow on the full dataset               | High       | High   | Histogram binning, vectorized `bincount` split search, subsample tuning, early benchmark on 100K rows, cap model size |
| Phase 3 overruns the schedule                                  | High       | High   | Shared tree builder, cut order in section 5, ship LR + RF + one booster as the fallback                               |
| Model looks great offline but fails on real traffic (C3)       | High       | High   | Group and time splits, realistic benign set, out-of-time hold-out                                                     |
| High false positive rate annoys users                          | High       | High   | Allowlist, strict threshold tied to FPR target, easy override and reporting                                           |
| Privacy complaint or store rejection (C2)                      | Medium     | High   | Sanitize URLs, consent screen, minimal permissions, on-device mode, accurate policy                                   |
| Python and JavaScript feature extractors disagree (M12)        | Medium     | Medium | Shared golden test cases run in CI                                                                                    |
| Backend latency makes browsing feel slow                       | Medium     | Medium | Local cache, allowlist, on-device first pass, regional hosting                                                        |
| Training data becomes stale                                    | High       | Medium | Scheduled retraining, blocklist feeds, drift monitoring                                                               |
| API abuse or scraping                                          | Medium     | Medium | Rate limits, install tokens, monitoring                                                                               |
| Report endpoint poisoned by attackers                          | Medium     | Medium | Human review before training use                                                                                      |
| Legal or compliance misread                                    | Medium     | High   | Qualified review of policy and terms before launch                                                                    |
| Scope creep (content features, multi-class, serverless)        | High       | Medium | Stick to the MVP scope in section 1                                                                                   |

---

## 9. Definition of Done (MVP)

- [ ] All models, calibration, and metrics implemented using only NumPy, pandas, and matplotlib (plus the standard library), with unit tests passing.
- [ ] Calibrated model meets the agreed FPR and recall targets on the out-of-time hold-out.
- [ ] Evaluation figures (ROC, PR, calibration, confusion matrix, feature importance) generated with matplotlib and included in the model card.
- [ ] API deployed over HTTPS with rate limiting, caching, and no raw URL logging.
- [ ] Extension installs, shows consent, classifies navigation, warns on unsafe URLs, and fails safely.
- [ ] Website live with Home, How It Works, Privacy Policy, About, Contact, and Terms, all accurate.
- [ ] Compliance checklist complete and policy reviewed.
- [ ] Model card, data card, and threat model written.
- [ ] Monitoring and a rollback plan in place.
