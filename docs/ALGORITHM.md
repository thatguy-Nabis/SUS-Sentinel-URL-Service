# URL Classification Algorithm

This document describes the URL preparation, feature extraction, model training, and request-time behavior implemented in this repository. It distinguishes the training pipeline from the model that the API actually runs, since those paths currently differ.

## 1. Task and labels

The trained models solve a binary classification problem:

| Label | Meaning |
| --- | --- |
| `0` | Benign / legitimate |
| `1` | Malicious / unsafe |

During data preparation, `benign`, `legitimate`, and `good` map to `0`; `malicious`, `phishing`, `malware`, `defacement`, `spam`, `exploit`, and `bad` map to `1`. Labels not in this mapping are discarded. In particular, there is no separate adult-content class. The API's `category` field is always `null`; the model does not predict phishing, malware, spam, or other subclasses.

All model and feature code is implemented with NumPy and Python's standard library (pandas is used for CSV preparation). The feature set is lexical: inference does not fetch a page, perform DNS or WHOIS lookups, or inspect HTML or JavaScript.

## 2. Data preparation

`ml/training/data_prep.py` loads whichever of the configured balanced URL, MURL, and DMOZ CSV files exist. DMOZ rows are treated as benign and sampled up to 50,000 in the normal preparation flow. After label mapping, URLs are normalized with query strings and fragments removed, short/empty URLs are discarded, and duplicate normalized URLs are dropped (the first row wins).

If the dataset exceeds the requested subsample size, preparation samples up to half from each class, then splits the unique registered-domain groups: 70% of groups to train, 15% to validation, and the remainder to test. These are domain-group shares, not guaranteed row shares. Every URL from a given parsed registered domain stays in one split. Domain parsing uses a small hard-coded list of common two-part public suffixes; it does not use the Public Suffix List. The split is randomized, not time-based. Conflicting duplicate labels are not reconciled before keeping the first normalized URL.

The resulting `train.csv`, `val.csv`, and `test.csv` contain URL and binary label columns (plus any source column retained from ingestion). Training extracts features from each split. `--max_samples` on the trainer samples each split independently with seed 42.

## 3. URL normalization and parsing

`normalize_url` trims whitespace, adds `http://` when no `scheme://` prefix exists (`//host` becomes `http://host`), lowercases the scheme and network location, removes default HTTP/HTTPS ports, supplies `/` for an empty path, and collapses repeated slashes in the path. Its default mode removes the query and fragment. Feature extraction itself requests the mode that preserves them, but both training data preparation and the API clean URLs first, so their main model inputs do not contain query strings or fragments.

`extract_domain_info` derives the host, registered-domain approximation, subdomain, TLD, path, scheme, and IP indicator. Its IP helper recognizes IPv4 only when all four decimal octets are in 0–255 and uses a broad hexadecimal/colon pattern for IPv6. However, domain parsing first splits the network location on `:`, so ordinary bracketed IPv6 hosts can be truncated before the helper sees them. For domains, it treats entries such as `co.uk` and `com.np` as two-part suffixes only when they appear in its fixed list. Otherwise, the last label is the TLD and the last two labels form the registered domain. These are deterministic string heuristics, not authoritative public-suffix or DNS checks.

## 4. Feature extraction

`LexicalFeatureExtractor` emits 30 floating-point features in this exact order. Counts and ratios are computed from the normalized, lowercased URL unless stated otherwise.

| # | Feature | Definition |
| ---: | --- | --- |
| 1 | `url_length` | Number of characters in the normalized URL. |
| 2 | `host_length` | Number of host characters, excluding port. |
| 3 | `path_length` | Number of path characters; an empty path is `/`. |
| 4 | `has_https` | 1 if the parsed scheme is HTTPS, otherwise 0. |
| 5 | `is_ip` | 1 if the parsed host matches the IP rules above. |
| 6 | `is_shortener` | 1 if host or registered domain exactly matches one of the configured shortener domains. |
| 7–16 | `count_dots`, `count_hyphens`, `count_underscores`, `count_slashes`, `count_question_marks`, `count_equal_signs`, `count_at_symbols`, `count_percent_signs`, `count_ampersands`, `count_digits` | Counts of the corresponding character (digits count all `str.isdigit()` characters). |
| 17 | `digit_ratio_url` | URL digit count divided by URL length, with denominator at least 1. |
| 18 | `digit_ratio_host` | Host digit count divided by host length, with denominator at least 1. |
| 19 | `subdomain_count` | Number of dot-separated subdomain labels, or 0 if absent. |
| 20 | `url_entropy` | Shannon character entropy in bits over the normalized URL. |
| 21 | `host_entropy` | Shannon character entropy in bits over the host. |
| 22 | `suspicious_keyword_count` | Number of distinct configured authentication/deception keywords appearing anywhere in the URL. |
| 23 | `suspicious_keyword_in_host` | 1 if any such keyword appears in the host. |
| 24 | `high_risk_tld` | 1 if the parsed TLD is in the fixed high-risk TLD set. |
| 25 | `brand_in_subdomain_or_path` | 1 if a configured brand appears in subdomain or path and does not appear in the parsed registered domain. |
| 26 | `has_double_slash_in_path` | 1 if `//` occurs in the parsed path. |
| 27 | `longest_token_length` | Length of the longest URL token split on non-word characters and underscores. |
| 28 | `vowel_consonant_ratio` | Count of `a/e/i/o/u` divided by other alphabetic characters in the URL; denominator at least 1. |
| 29 | `special_char_ratio` | Non-alphanumeric URL characters divided by URL length; denominator at least 1. |
| 30 | `has_port_specified` | 1 if the parsed network location contains `:`. |

The keyword, TLD, shortener, and brand lists are constants in `ml/features/extractor.py`. Brand checks are substring checks (for example, a brand string in a path), not edit-distance or visual-similarity checks. These signals are model inputs, not standalone verdict rules.

Shannon entropy for a string of length `n` is `-sum(p(c) * log2(p(c)))` over its distinct characters. An empty string has entropy 0. The configured string lists are:

- Suspicious keywords: `login`, `verify`, `update`, `security`, `account`, `banking`, `secure`, `webscr`, `signin`, `confirm`, `ebayisapi`, `password`, `authenticate`, `wallet`, `admin`, `service`, `support`, `billing`, `recover`, `validate`.
- High-risk TLDs: `xyz`, `top`, `club`, `work`, `ru`, `cn`, `tk`, `ml`, `ga`, `cf`, `gq`, `buzz`, `fit`, `rest`, `surf`, `monster`, `icu`, `cam`.
- Shorteners: `bit.ly`, `tinyurl.com`, `t.co`, `goo.gl`, `is.gd`, `buff.ly`, `ow.ly`, `rebrand.ly`, `cutt.ly`, `tiny.cc`, `bit.do`, `shorte.st`.
- Brands checked for spoofing: `paypal`, `google`, `apple`, `microsoft`, `amazon`, `netflix`, `chase`, `facebook`, `instagram`, `whatsapp`, `wellsfargo`, `binance`.

An important serving detail: query and fragment text is stripped before inference, so the query-related character counts (question mark, equals, ampersand, and percent) are normally zero in API predictions. Training URLs are also query-stripped by data preparation, keeping the usual training and serving URL forms aligned on this point.

## 5. Models implemented by the training code

### 5.1 Logistic regression

The trainer fits a `CustomStandardScaler` on training features only. Each feature becomes `(x - mean) / scale`; a constant feature uses scale 1. Logistic regression is trained on these scaled features. It initializes weights and bias to zero and minimizes binary cross-entropy with L2 regularization using shuffled mini-batches and Adam (`beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`). The positive class is malicious. The batch gradients are:

```text
p = sigmoid(X @ weights + bias)
error = (p - y) * sample_weight
dw = X.T @ error / batch_size + lambda * weights
db = sum(error) / batch_size
```

With `class_weight="balanced"`, each class receives total weight approximately one half: `n / (2 * class_count)`. The trainer uses learning rate 0.03, L2 coefficient 0.005, at most 40 epochs, batch size 1024, and balanced class weights. It stops when the absolute change in its weighted training loss is below `1e-5`. Validation loss is recorded but does not control logistic-regression stopping.

The model's raw score is the linear logit `X @ weights + bias`; its uncalibrated positive probability is the sigmoid of that score.

### 5.2 Histogram CART and random forest

`BinMapper` computes per-feature quantile edges from training data for 64 bins. Each feature value is converted to a compact integer bin using `numpy.digitize`. At a tree node, candidate thresholds are evaluated from cumulative positive and negative counts per bin. The split maximizing parent Gini impurity minus weighted child Gini impurity is selected, subject to minimum split and leaf sample counts. A leaf returns its fraction of positive examples. Trees are stored as flat arrays of feature indices, bin thresholds, child indices, and leaf probabilities.

The random forest fits one shared bin mapper, then trains bootstrap samples (sample count equal to the original training size) with a random feature subset at each node. The trainer configures 20 trees, maximum depth 7, minimum split 20, minimum leaf 10, and `sqrt(number_of_features)` candidate features. The predicted probability is the average of tree leaf probabilities. Out-of-bag predictions for samples omitted from each bootstrap sample are averaged and used to report an OOB accuracy at threshold 0.5.

### 5.3 Histogram gradient boosting

`GradientBoostingClassifier` is binary logistic gradient boosting over binned features. Its initial raw score is the log-odds of the training positive rate, clipped to a rate between `1e-4` and `1 - 1e-4`. At each boosting round it computes `p = sigmoid(raw_score)`, gradient `g = p - y`, and Hessian `h = max(p * (1 - p), 1e-8)`. It then fits a small regression tree to those gradient/Hessian statistics and adds `learning_rate * tree_output` to the raw score.

For a candidate split, the gain is:

```text
0.5 * (G_left^2 / (H_left + L2)
     + G_right^2 / (H_right + L2)
     - G_parent^2 / (H_parent + L2))
```

Each leaf value is `-G / (H + L2)`. The implementation supports a depth-limited recursive (`depthwise`) growth policy and a best-gain leaf-wise (`lossguide`) policy. The current trainer uses only `depthwise`; it does not instantiate a LightGBM-style leaf-wise model. Training uses 30 maximum rounds, learning rate 0.1, maximum depth 5, row subsampling 0.8 without replacement, feature subsampling 0.8 per tree, and L2 / minimum-child-weight defaults of 1.0. Validation log loss is monitored; after eight rounds without an improvement of at least `1e-4`, training stops. The implementation stops at that point but does not truncate the ensemble back to its best validation round.

The model's probability is the sigmoid of the accumulated raw score. This is a from-scratch gradient-boosting implementation with XGBoost-style second-order split scoring, not the XGBoost or LightGBM libraries.

## 6. Training ensemble and probability calibration

The trainer combines validation probabilities from logistic regression, random forest, and gradient boosting using weights `[0.25, 0.35, 0.40]`, respectively. Although `WeightedEnsemble.fit_weights` contains a small validation grid search, the training script does not call it; it assigns those weights directly. The raw ensemble probability is:

```text
p_blend = 0.25 * p_logistic + 0.35 * p_forest + 0.40 * p_boosting
```

For calibration, the blend is clipped to `[1e-6, 1 - 1e-6]`, converted to log-odds, and passed to a Platt calibrator. Platt scaling fits `sigmoid(a * score + b)` on the validation labels with Newton updates to logistic loss and smoothed positive/negative targets. This calibrated blend is evaluated on the held-out test split. An isotonic (PAVA) calibrator is also implemented in `ml/algorithms/calibration.py`, but the training script does not use it.

The training script reports accuracy, precision, recall, F1, false-positive rate, ROC-AUC, PR-AUC, recall at false-positive rate at most 0.1%, and expected calibration error. The current `ml/models/metadata.json` records accuracy 0.7815, precision 0.7470, recall 0.5695, F1 0.6463, ROC-AUC 0.8595, PR-AUC 0.7841, recall-at-0.1%-FPR 0.0976, and ECE 0.0356. Those are the training script's ensemble evaluation values; see the deployment mismatch below before treating them as live API performance.

## 7. What the API actually runs

`backend/app/model.py` loads `ml/models/classifier_v1.npz`. That artifact contains the logistic-regression weights and bias, scaler mean and scale, Platt `a` and `b`, and the ensemble-weight array. It does **not** contain the trained forest or boosting trees. The backend reads only the logistic-regression weights, scaler, and Platt values; it ignores the saved ensemble weights.

For a non-allowlisted request, the actual loaded-model calculation is:

```text
clean_url = normalize_url(url, strip_query_and_fragment=True)
x = extract_30_features(clean_url)
x_scaled = (x - scaler_mean) / scaler_scale
logit = dot(x_scaled, logistic_weights) + logistic_bias
confidence = sigmoid(platt_a * logit + platt_b)
```

This means the live API is a Platt-transformed logistic-regression score, not the calibrated three-model blend measured by the trainer. The trainer fits its calibrator to the logit of the *blended probability*, while the backend applies the same `a` and `b` directly to the logistic model's raw logit. The saved ensemble weights do not change live predictions. Consequently, the ensemble metrics in metadata are not a direct evaluation of the current API path, and `calibrated: true` in the API response should be read as an implementation flag rather than proof that this deployed score was calibrated on the same score distribution.

If loading the artifact fails, the engine uses a hand-built baseline score instead: start at 0.05, add 0.45 for an IP host, 0.35 for a high-risk TLD, 0.40 for a brand in subdomain/path, 0.25 for a suspicious keyword in the host, and 0.15 when HTTPS is absent; cap at 0.98. This fallback is not the trained model. The response currently still labels its source as `model` and `calibrated` as `true`.

## 8. API decision flow and verdicts

For `POST /v1/classify`, the service trims the submitted URL, rejects an empty value, normalizes it with query and fragment removal, and then:

1. Checks the in-memory seed allowlist. A match returns `safe`, confidence `0.01`, and source `allowlist` without running the model. Registered-domain or host equality and subdomain suffixes are accepted.
2. Checks an in-memory prediction cache keyed by SHA-256 of the cleaned URL. Entries live for 3,600 seconds; the cache holds up to 10,000 entries and evicts the entry with the oldest timestamp when full.
3. Runs the inference calculation above and attaches up to three deterministic explanation strings from lexical rules.

The loaded-model confidence maps to verdicts using the configured thresholds:

| Confidence | Verdict |
| --- | --- |
| `< 0.30` | `safe` |
| `0.30` to `< 0.70` | `suspicious` |
| `>= 0.70` | `unsafe` |

The API returns the normalized URL, verdict, null category, confidence, `calibrated`, source, model version, reasons, latency, and cached flag. These three verdict bands differ from the binary 0.5 threshold used for model evaluation. Metadata's `strict_recall_threshold` is recorded for analysis but is not used for API verdicts.

The browser extension calls the API after an HTTP(S) tab finishes loading. It sends protocol, hostname, and path (dropping query, fragment, credentials, and port in its URL reconstruction), then displays a SAFE, SUSP, or RISK badge. If the request fails, it displays OFF and does not block navigation.

## 9. Explanation rules

The returned reasons are not feature attributions from logistic-regression weights. They are a separate ordered set of simple checks: raw IP host; brand string in subdomain/path; high-risk TLD; suspicious keyword in host; known shortener; no HTTPS; at least three hyphens; host digit ratio above 0.3; at least three subdomain labels; URL entropy above 4.5. The first three matches are returned. If none match, the service returns “No common lexical risk indicators detected.” A URL can therefore receive a model verdict without a matching explanation rule, or show a reason that is not the model's strongest contributing feature.

## 10. Artifacts and implementation boundaries

`ml/models/classifier_v1.npz` is the runtime parameter archive and `ml/models/metadata.json` records feature names, model version, training metrics, thresholds, and calibrator parameters. The trainer's serialization currently saves logistic-regression parameters and metadata, not the random-forest or boosting structures. Multi-class output, live reputation data, page-content features, and browser-side inference are not implemented in this code path.

Primary implementation files:

- `ml/features/normalizer.py` — URL normalization and domain parsing.
- `ml/features/extractor.py` — 30 lexical features and explanation rules.
- `ml/algorithms/logistic_regression.py` — scaler and logistic regression.
- `ml/algorithms/tree.py` and `ml/algorithms/random_forest.py` — histogram CART and forest.
- `ml/algorithms/boosting.py` — second-order histogram boosting.
- `ml/algorithms/ensemble.py` and `ml/algorithms/calibration.py` — blending and calibration helpers.
- `ml/training/data_prep.py` and `ml/training/train.py` — dataset preparation, fitting, evaluation, and serialization.
- `backend/app/model.py` and `backend/app/main.py` — runtime inference and API decision flow.
