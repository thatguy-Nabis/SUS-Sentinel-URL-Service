# Dataset Card: Unified Unambiguous URL Security Dataset

## 1. Overview
This dataset combines, cleans, standardizes, and unifies the three major URL security corpora into a single **unambiguous, deduplicated ground truth dataset** for training URL classification models.

- **Storage Location:** `Dataset/Dataset-Filtered/`
- **Total Clean Instances:** 1,209,827
- **Safe / Benign:** 805,158 (66.55%)
- **Malicious / Unsafe:** 404,669 (33.45%)
- **Unique Registered Domains:** 291,223
- **Domain Overlap Between Splits:** **0 (Zero Domain Leakage)**

---

## 2. Ingested Data Sources
1. **GitHub Malicious URL Detection Dataset** (`balanced_urls.csv`):
   - Provenance: Balanced collection of verified benign and malicious URLs.
2. **Mendeley Data MURL Dataset** (`Malicious URL Detection.csv`):
   - Provenance: Multi-class security research dataset covering Phishing, Malware, Defacement, Legitimate, Spam, and Exploit.
3. **Zenodo Webpage Topic & Threat Dataset** (`phishstorm.csv` & `dmoz.csv`):
   - Provenance: PhishStorm phishing research feed + Open Directory Project (DMOZ) legitimate directory URLs.
   - *Ambiguity Resolution:* DMOZ 'Adult' content policy URLs were filtered out to avoid confounding content policy categories with malware/phishing threats.

---

## 3. Ambiguity Resolution & Conflict Filtering
During cross-dataset unification, identical canonical URLs can appear with conflicting labels due to different crawl timeframes or noisy feeds.
- **Strict Consensus Policy:** Any canonical URL with conflicting labels across datasets (168,413 instances) was **quarantined and dropped** from training to ensure zero label contradiction in model optimization.
- **Audit Log:** Dropped conflicting instances are logged in `Dataset/Dataset-Filtered/label_conflicts.csv`.

---

## 4. Fine-Grained Category Breakdown
| Category | Count | Percentage |
| :--- | :--- | :--- |
| `benign` | 805,158 | 66.55% |
| `phishing` | 227,204 | 18.78% |
| `malicious` | 107,336 | 8.87% |
| `defacement` | 44,508 | 3.68% |
| `malware` | 22,657 | 1.87% |
| `spam` | 2,721 | 0.22% |
| `exploit` | 243 | 0.02% |

---

## 5. Domain-Group Partitioning
To prevent the common artifact of models memorizing popular domains across train and test sets, partitioning is executed strictly by **registered domain grouping**:
- `train.csv`: 816,165 rows (70% of domain groups)
- `val.csv`: 187,855 rows (15% of domain groups)
- `test.csv`: 205,807 rows (15% of domain groups)
- **Domain Leakage:** Confirmed 0 overlapping domains.
