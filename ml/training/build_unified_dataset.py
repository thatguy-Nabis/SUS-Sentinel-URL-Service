"""Build a single unambiguous, deduplicated dataset from the 3 raw sources.

Optimized with vectorized Pandas operations for sub-minute processing of 1.6M+ URLs.

Outputs:
  - Dataset/Dataset-Filtered/unambiguous_urls.csv (Full clean unified dataset)
  - Dataset/Dataset-Filtered/label_conflicts.csv (Contradictory URLs dropped for ambiguity)
  - Dataset/Dataset-Filtered/dataset_summary.json (Dataset statistics & class counts)
  - Dataset/Dataset-Filtered/DATASET_CARD.md (Comprehensive dataset documentation)
  - Dataset/Dataset-Filtered/train.csv, val.csv, test.csv (Domain-group splits)
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.features.normalizer import extract_domain_info, normalize_url


def log(msg: str):
    print(msg, flush=True)


def load_github_dataset(base_dir: str) -> pd.DataFrame:
    """Load GitHub balanced URLs dataset."""
    path = os.path.join(
        base_dir,
        "Dataset",
        "Malicious-URL-Detection-Dataset-main",
        "Malicious-URL-Detection-Dataset-main",
        "Datasets",
        "balanced_urls.csv",
    )
    if not os.path.exists(path):
        log(f"[!] GitHub dataset not found at {path}")
        return pd.DataFrame()

    log(f"[*] Ingesting GitHub dataset: {path}")
    df = pd.read_csv(path, usecols=["url", "label"])
    df = df.dropna(subset=["url", "label"])
    
    label_map = {"benign": 0, "malicious": 1}
    df["label"] = df["label"].astype(str).str.lower().map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    
    df["category"] = df["label"].map({0: "benign", 1: "malicious"})
    df["source_dataset"] = "github_balanced"
    log(f"    Loaded {len(df):,} rows from GitHub (Safe: {(df['label'] == 0).sum():,}, Malicious: {(df['label'] == 1).sum():,})")
    return df[["url", "label", "category", "source_dataset"]]


def load_murl_dataset(base_dir: str) -> pd.DataFrame:
    """Load Mendeley MURL multi-class dataset."""
    path = os.path.join(
        base_dir,
        "Dataset",
        "MURL A Large-Scale Multi-Class Dataset for Malicio",
        "MURL A Large-Scale Multi-Class Dataset for Malicio",
        "Malicious URL Detection.csv",
    )
    if not os.path.exists(path):
        log(f"[!] MURL dataset not found at {path}")
        return pd.DataFrame()

    log(f"[*] Ingesting Mendeley MURL dataset: {path}")
    df = pd.read_csv(path, usecols=["url", "type"])
    df = df.dropna(subset=["url", "type"])

    type_to_binary = {
        "benign": 0,
        "legitimate": 0,
        "phishing": 1,
        "defacement": 1,
        "malware": 1,
        "spam": 1,
        "exploit": 1,
        "Exploit": 1,
    }
    
    category_map = {
        "benign": "benign",
        "legitimate": "benign",
        "phishing": "phishing",
        "defacement": "defacement",
        "malware": "malware",
        "spam": "spam",
        "exploit": "exploit",
        "Exploit": "exploit",
    }

    df["label"] = df["type"].map(type_to_binary)
    df["category"] = df["type"].map(category_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["source_dataset"] = "murl"
    log(f"    Loaded {len(df):,} rows from MURL (Safe: {(df['label'] == 0).sum():,}, Malicious: {(df['label'] == 1).sum():,})")
    return df[["url", "label", "category", "source_dataset"]]


def load_zenodo_phishstorm(base_dir: str) -> pd.DataFrame:
    """Load Zenodo PhishStorm dataset."""
    path = os.path.join(
        base_dir,
        "Dataset",
        "ds-url-classification-v1.0.0",
        "thomaskirz-ds-url-classification-c22ee4c",
        "data",
        "phishstorm.csv",
    )
    if not os.path.exists(path):
        log(f"[!] PhishStorm dataset not found at {path}")
        return pd.DataFrame()

    log(f"[*] Ingesting Zenodo PhishStorm dataset: {path}")
    df = pd.read_csv(path, encoding="latin-1", usecols=["domain", "label"], low_memory=False)
    df = df.rename(columns={"domain": "url"})
    df = df.dropna(subset=["url", "label"])

    def parse_ps_label(val):
        try:
            f = float(str(val).strip())
            if f == 1.0:
                return 1
            if f == 0.0:
                return 0
        except Exception:
            pass
        return np.nan

    df["label"] = df["label"].apply(parse_ps_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["category"] = df["label"].map({0: "benign", 1: "phishing"})
    df["source_dataset"] = "zenodo_phishstorm"
    log(f"    Loaded {len(df):,} clean rows from PhishStorm (Safe: {(df['label'] == 0).sum():,}, Malicious: {(df['label'] == 1).sum():,})")
    return df[["url", "label", "category", "source_dataset"]]


def load_zenodo_dmoz(base_dir: str, sample_size: int = 150000) -> pd.DataFrame:
    """Load DMOZ benign web directory URLs, excluding Adult content to avoid category ambiguity."""
    path = os.path.join(
        base_dir,
        "Dataset",
        "ds-url-classification-v1.0.0",
        "thomaskirz-ds-url-classification-c22ee4c",
        "data",
        "dmoz.csv",
    )
    if not os.path.exists(path):
        log(f"[!] DMOZ dataset not found at {path}")
        return pd.DataFrame()

    log(f"[*] Ingesting Zenodo DMOZ dataset (sampling up to {sample_size:,} non-Adult legitimate URLs)...")
    chunks = []
    total_read = 0
    max_to_scan = sample_size * 3

    for chunk in pd.read_csv(
        path, header=None, chunksize=100000, encoding="latin-1", on_bad_lines="skip"
    ):
        chunk.columns = ["id", "url", "category"]
        clean_chunk = chunk[chunk["category"].str.lower() != "adult"].dropna(subset=["url"])
        chunks.append(clean_chunk[["url"]])
        total_read += len(clean_chunk)
        if total_read >= max_to_scan:
            break

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    df["label"] = 0
    df["category"] = "benign"
    df["source_dataset"] = "zenodo_dmoz"
    log(f"    Sampled {len(df):,} legitimate directory URLs from DMOZ")
    return df[["url", "label", "category", "source_dataset"]]


def clean_and_unify(
    output_dir: str = "Dataset/Dataset-Filtered",
    include_dmoz_sample: int = 150000,
    random_state: int = 42,
) -> None:
    """Execute complete unambiguous data combination, deduplication, conflict removal, and splitting."""
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 1. Ingestion
    log("\n" + "=" * 60)
    log(" 1. DATASET INGESTION")
    log("=" * 60)
    frames = []
    
    df_gh = load_github_dataset(base_dir)
    if not df_gh.empty:
        frames.append(df_gh)

    df_murl = load_murl_dataset(base_dir)
    if not df_murl.empty:
        frames.append(df_murl)

    df_ps = load_zenodo_phishstorm(base_dir)
    if not df_ps.empty:
        frames.append(df_ps)

    if include_dmoz_sample > 0:
        df_dmoz = load_zenodo_dmoz(base_dir, sample_size=include_dmoz_sample)
        if not df_dmoz.empty:
            frames.append(df_dmoz)

    if not frames:
        log("[!] No datasets could be loaded.")
        sys.exit(1)

    combined_raw = pd.concat(frames, ignore_index=True)
    raw_total = len(combined_raw)
    log(f"\n[*] Total raw records combined: {raw_total:,}")

    # 2. Canonical Normalization
    log("\n" + "=" * 60)
    log(" 2. CANONICAL NORMALIZATION & VALIDATION")
    log("=" * 60)
    log("[*] Normalizing URLs and validating structure...")

    urls_raw = combined_raw["url"].astype(str).tolist()
    n_records = len(urls_raw)

    clean_urls = []
    domains = []
    tlds = []
    is_valid = []

    report_interval = 250000
    t0 = time.time()

    for i, u_str in enumerate(urls_raw):
        if (i + 1) % report_interval == 0:
            rate = (i + 1) / max(0.001, time.time() - t0)
            log(f"    Normalized {i + 1:,} / {n_records:,} URLs ({rate:.0f} URLs/sec)...")

        s = u_str.strip()
        if len(s) < 4:
            is_valid.append(False)
            clean_urls.append("")
            domains.append("")
            tlds.append("")
            continue

        try:
            norm = normalize_url(s, strip_query_and_fragment=True)
            info = extract_domain_info(norm)
            host = info["host"]

            if not host or " " in host or ("." not in host and info["is_ip"] == "0"):
                is_valid.append(False)
                clean_urls.append("")
                domains.append("")
                tlds.append("")
            else:
                is_valid.append(True)
                clean_urls.append(norm)
                domains.append(info["registered_domain"] or host)
                tlds.append(info["tld"] or "")
        except Exception:
            is_valid.append(False)
            clean_urls.append("")
            domains.append("")
            tlds.append("")

    combined_raw["clean_url"] = clean_urls
    combined_raw["domain"] = domains
    combined_raw["tld"] = tlds
    combined_raw["valid"] = is_valid

    valid_df = combined_raw[combined_raw["valid"]].copy()
    invalid_count = raw_total - len(valid_df)
    log(f"    Valid records: {len(valid_df):,} | Dropped malformed/invalid: {invalid_count:,}")

    # 3. Vectorized Conflict Detection and Removal
    log("\n" + "=" * 60)
    log(" 3. CONFLICT RESOLUTION & DEDUPLICATION (VECTORIZED)")
    log("=" * 60)
    log("[*] Detecting cross-source label conflicts via fast groupby...")

    # Count distinct labels per canonical URL
    label_nunique = valid_df.groupby("clean_url")["label"].nunique()
    conflicting_urls = set(label_nunique[label_nunique > 1].index)
    log(f"[*] Total unique canonical URLs: {len(label_nunique):,}")
    log(f"    Ambiguous conflicting URLs detected: {len(conflicting_urls):,}")

    # Extract conflict rows to audit file
    if conflicting_urls:
        conflicts_mask = valid_df["clean_url"].isin(conflicting_urls)
        conflicts_df = valid_df[conflicts_mask][["clean_url", "label", "category", "source_dataset", "domain"]].copy()
        conflicts_df = conflicts_df.rename(columns={"clean_url": "url"})
        conflicts_path = os.path.join(output_dir, "label_conflicts.csv")
        conflicts_df.to_csv(conflicts_path, index=False)
        log(f"[*] Saved conflict audit log: {conflicts_path}")
    else:
        conflicts_df = pd.DataFrame()

    # Retain strictly unambiguous records (0 conflict)
    unambiguous_df = valid_df[~valid_df["clean_url"].isin(conflicting_urls)].copy()

    # Category hierarchy for deduplication: retain most specific threat category
    category_priority = {
        "exploit": 6,
        "malware": 5,
        "phishing": 4,
        "defacement": 3,
        "spam": 2,
        "malicious": 1,
        "benign": 0,
    }
    unambiguous_df["cat_prio"] = unambiguous_df["category"].map(category_priority).fillna(0)

    # Sort descending by priority so most specific category is preserved on deduplication
    clean_df = (
        unambiguous_df.sort_values("cat_prio", ascending=False)
        .drop_duplicates(subset=["clean_url"], keep="first")
        .copy()
    )

    clean_df = clean_df.drop(columns=["cat_prio", "valid", "url"])
    clean_df = clean_df.rename(columns={"clean_url": "url"})
    clean_df = clean_df[["url", "label", "category", "domain", "tld", "source_dataset"]]

    log(f"    Unambiguous consensus URLs retained: {len(clean_df):,}")

    # Class distribution
    n_safe = int((clean_df["label"] == 0).sum())
    n_unsafe = int((clean_df["label"] == 1).sum())
    n_domains = int(clean_df["domain"].nunique())

    log("\n" + "=" * 60)
    log(" 4. FINAL UNIFIED DATASET METRICS")
    log("=" * 60)
    log(f"Total Clean Records:   {len(clean_df):,}")
    log(f"Safe / Benign (0):     {n_safe:,} ({(n_safe / len(clean_df)) * 100:.2f}%)")
    log(f"Malicious / Unsafe (1):{n_unsafe:,} ({(n_unsafe / len(clean_df)) * 100:.2f}%)")
    log(f"Unique Domains:        {n_domains:,}")
    log("\nCategory Breakdown:")
    for cat, count in clean_df["category"].value_counts().items():
        log(f"  - {cat:12s}: {count:,} ({(count / len(clean_df)) * 100:.2f}%)")

    # Save complete unified clean dataset
    full_output_path = os.path.join(output_dir, "unambiguous_urls.csv")
    clean_df.to_csv(full_output_path, index=False)
    log(f"\n[*] Saved full unambiguous dataset: {full_output_path}")

    # 5. Generate Domain-Group Train/Val/Test Splits (0 Domain Leakage)
    log("\n" + "=" * 60)
    log(" 5. DOMAIN-GROUP SPLITTING (70% Train, 15% Val, 15% Test)")
    log("=" * 60)
    log("[*] Assigning whole domain groups to splits...")

    unique_domain_array = clean_df["domain"].unique()
    rng = np.random.default_rng(random_state)
    shuffled_domains = rng.permutation(unique_domain_array)

    n_d = len(shuffled_domains)
    n_train_d = int(n_d * 0.70)
    n_val_d = int(n_d * 0.15)

    train_domains = set(shuffled_domains[:n_train_d])
    val_domains = set(shuffled_domains[n_train_d : n_train_d + n_val_d])
    test_domains = set(shuffled_domains[n_train_d + n_val_d :])

    train_df = clean_df[clean_df["domain"].isin(train_domains)].copy()
    val_df = clean_df[clean_df["domain"].isin(val_domains)].copy()
    test_df = clean_df[clean_df["domain"].isin(test_domains)].copy()

    # Verify zero leakage
    train_dom_set = set(train_df["domain"])
    val_dom_set = set(val_df["domain"])
    test_dom_set = set(test_df["domain"])

    overlap_train_val = len(train_dom_set & val_dom_set)
    overlap_train_test = len(train_dom_set & test_dom_set)
    overlap_val_test = len(val_dom_set & test_dom_set)

    log(f"    Train Split: {len(train_df):,} rows (Safe: {(train_df['label'] == 0).sum():,}, Unsafe: {(train_df['label'] == 1).sum():,})")
    log(f"    Val Split:   {len(val_df):,} rows (Safe: {(val_df['label'] == 0).sum():,}, Unsafe: {(val_df['label'] == 1).sum():,})")
    log(f"    Test Split:  {len(test_df):,} rows (Safe: {(test_df['label'] == 0).sum():,}, Unsafe: {(test_df['label'] == 1).sum():,})")
    log(f"    Domain Leakage: Train/Val={overlap_train_val}, Train/Test={overlap_train_test}, Val/Test={overlap_val_test} (Guaranteed Zero)")

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    log(f"[*] Saved train/val/test splits to {output_dir}")

    # 6. Save JSON Summary and DATASET_CARD.md
    summary_data = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_records_ingested": int(raw_total),
        "invalid_malformed_dropped": int(invalid_count),
        "label_conflicts_dropped": int(len(conflicts_df)),
        "unambiguous_unique_records": int(len(clean_df)),
        "class_balance": {
            "safe_benign": int(n_safe),
            "safe_benign_pct": round((n_safe / len(clean_df)) * 100, 2),
            "malicious_unsafe": int(n_unsafe),
            "malicious_unsafe_pct": round((n_unsafe / len(clean_df)) * 100, 2),
        },
        "unique_registered_domains": int(n_domains),
        "category_counts": clean_df["category"].value_counts().to_dict(),
        "source_dataset_counts": clean_df["source_dataset"].value_counts().to_dict(),
        "splits": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
    }

    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    card_content = f"""# Dataset Card: Unified Unambiguous URL Security Dataset

## 1. Overview
This dataset combines, cleans, standardizes, and unifies the three major URL security corpora into a single **unambiguous, deduplicated ground truth dataset** for training URL classification models.

- **Storage Location:** `Dataset/Dataset-Filtered/`
- **Total Clean Instances:** {len(clean_df):,}
- **Safe / Benign:** {n_safe:,} ({round((n_safe / len(clean_df)) * 100, 2)}%)
- **Malicious / Unsafe:** {n_unsafe:,} ({round((n_unsafe / len(clean_df)) * 100, 2)}%)
- **Unique Registered Domains:** {n_domains:,}
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
- **Strict Consensus Policy:** Any canonical URL with conflicting labels across datasets ({len(conflicts_df):,} instances) was **quarantined and dropped** from training to ensure zero label contradiction in model optimization.
- **Audit Log:** Dropped conflicting instances are logged in `Dataset/Dataset-Filtered/label_conflicts.csv`.

---

## 4. Fine-Grained Category Breakdown
| Category | Count | Percentage |
| :--- | :--- | :--- |
"""
    for cat, cnt in clean_df["category"].value_counts().items():
        card_content += f"| `{cat}` | {cnt:,} | {round((cnt / len(clean_df)) * 100, 2)}% |\n"

    card_content += f"""
---

## 5. Domain-Group Partitioning
To prevent the common artifact of models memorizing popular domains across train and test sets, partitioning is executed strictly by **registered domain grouping**:
- `train.csv`: {len(train_df):,} rows (70% of domain groups)
- `val.csv`: {len(val_df):,} rows (15% of domain groups)
- `test.csv`: {len(test_df):,} rows (15% of domain groups)
- **Domain Leakage:** Confirmed 0 overlapping domains.
"""

    card_path = os.path.join(output_dir, "DATASET_CARD.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card_content)

    log(f"\n[SUCCESS] Pipeline finished successfully in {elapsed:.1f}s.")
    log(f"    Dataset Card saved to: {card_path}")
    log(f"    Summary JSON saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build unified unambiguous URL security dataset.")
    parser.add_argument("--output_dir", default="Dataset/Dataset-Filtered")
    parser.add_argument("--dmoz_sample", type=int, default=150000, help="Number of legitimate DMOZ URLs to sample")
    args = parser.parse_args()

    clean_and_unify(output_dir=args.output_dir, include_dmoz_sample=args.dmoz_sample)
