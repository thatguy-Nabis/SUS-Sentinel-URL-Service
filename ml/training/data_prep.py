import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from ml.features.normalizer import extract_domain_info, normalize_url


LABEL_MAP = {
    # Benign mappings -> 0
    "benign": 0,
    "legitimate": 0,
    "good": 0,
    # Malicious mappings -> 1
    "malicious": 1,
    "phishing": 1,
    "malware": 1,
    "defacement": 1,
    "spam": 1,
    "exploit": 1,
    "bad": 1,
}


def load_balanced_urls(path: str) -> pd.DataFrame:
    """Load balanced_urls.csv dataset."""
    if not os.path.exists(path):
        return pd.DataFrame()
    print(f"Loading {path}...")
    df = pd.read_csv(path, usecols=["url", "label"])
    df["label"] = df["label"].astype(str).str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)
    df["source"] = "balanced_urls"
    return df


def load_murl(path: str) -> pd.DataFrame:
    """Load MURL dataset."""
    if not os.path.exists(path):
        return pd.DataFrame()
    print(f"Loading {path}...")
    df = pd.read_csv(path, usecols=["url", "type"])
    df["label"] = df["type"].astype(str).str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)
    df["source"] = "murl"
    return df[["url", "label", "source"]]


def load_dmoz(path: str, max_samples: int = 200000) -> pd.DataFrame:
    """Sample benign directory URLs from DMOZ dataset."""
    if not os.path.exists(path):
        return pd.DataFrame()
    print(f"Loading DMOZ (sampling up to {max_samples})...")
    try:
        df = pd.read_csv(path, header=None, usecols=[1], nrows=max_samples * 2, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, header=None, usecols=[1], nrows=max_samples * 2, encoding="latin-1", on_bad_lines="skip")
    df.columns = ["url"]
    df["label"] = 0  # Benign
    df["source"] = "dmoz"
    df = df.dropna(subset=["url"])
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    return df


def domain_group_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset ensuring ZERO domain overlap between splits (domain-group split)."""
    print("Extracting registered domains for group split...")
    domains = []
    for u in df["url"]:
        info = extract_domain_info(u)
        domains.append(info["registered_domain"] or "unknown")
    df["domain"] = domains

    unique_domains = df["domain"].unique()
    rng = np.random.default_rng(random_state)
    shuffled_domains = rng.permutation(unique_domains)

    n_domains = len(shuffled_domains)
    n_train = int(n_domains * train_ratio)
    n_val = int(n_domains * val_ratio)

    train_domains = set(shuffled_domains[:n_train])
    val_domains = set(shuffled_domains[n_train:n_train + n_val])
    test_domains = set(shuffled_domains[n_train + n_val:])

    train_df = df[df["domain"].isin(train_domains)].copy()
    val_df = df[df["domain"].isin(val_domains)].copy()
    test_df = df[df["domain"].isin(test_domains)].copy()

    # Drop temporary domain column
    train_df = train_df.drop(columns=["domain"])
    val_df = val_df.drop(columns=["domain"])
    test_df = test_df.drop(columns=["domain"])

    return train_df, val_df, test_df


def prepare_data(
    output_dir: str = "data/processed",
    subsample_total: int = 100000,
    random_state: int = 42,
) -> None:
    """Ingest datasets, deduplicate, group-split, and save processed CSV splits."""
    os.makedirs(output_dir, exist_ok=True)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ds1_path = os.path.join(
        base_dir,
        "Dataset",
        "Malicious-URL-Detection-Dataset-main",
        "Malicious-URL-Detection-Dataset-main",
        "Datasets",
        "balanced_urls.csv",
    )
    ds2_path = os.path.join(
        base_dir,
        "Dataset",
        "MURL A Large-Scale Multi-Class Dataset for Malicio",
        "MURL A Large-Scale Multi-Class Dataset for Malicio",
        "Malicious URL Detection.csv",
    )
    ds3_path = os.path.join(
        base_dir,
        "Dataset",
        "ds-url-classification-v1.0.0",
        "thomaskirz-ds-url-classification-c22ee4c",
        "data",
        "dmoz.csv",
    )

    frames = []
    if os.path.exists(ds1_path):
        frames.append(load_balanced_urls(ds1_path))
    if os.path.exists(ds2_path):
        frames.append(load_murl(ds2_path))
    if os.path.exists(ds3_path):
        frames.append(load_dmoz(ds3_path, max_samples=50000))

    if not frames:
        print("Error: No raw datasets found.")
        sys.exit(1)

    print("Concatenating datasets...")
    full_df = pd.concat(frames, ignore_index=True)
    print(f"Total raw records: {len(full_df)}")

    print("Normalizing URLs...")
    full_df["url"] = full_df["url"].apply(lambda u: normalize_url(str(u), strip_query_and_fragment=True))
    full_df = full_df[full_df["url"].str.len() > 3]

    print("Deduplicating by normalized URL...")
    full_df = full_df.drop_duplicates(subset=["url"], keep="first")
    print(f"Total deduplicated records: {len(full_df)}")
    print(f"Class breakdown:\n{full_df['label'].value_counts()}")

    if subsample_total and len(full_df) > subsample_total:
        print(f"Subsampling to {subsample_total} samples with stratified class balance...")
        # Stratified subsample
        pos_df = full_df[full_df["label"] == 1]
        neg_df = full_df[full_df["label"] == 0]
        n_half = subsample_total // 2
        n_pos = min(len(pos_df), n_half)
        n_neg = min(len(neg_df), subsample_total - n_pos)

        sample_pos = pos_df.sample(n=n_pos, random_state=random_state)
        sample_neg = neg_df.sample(n=n_neg, random_state=random_state)
        full_df = pd.concat([sample_pos, sample_neg], ignore_index=True)
        print(f"Sampled class breakdown:\n{full_df['label'].value_counts()}")

    print("Performing domain-group split (70% train, 15% val, 15% test)...")
    train_df, val_df, test_df = domain_group_split(
        full_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=random_state
    )

    print(f"Train set: {len(train_df)} rows, Pos: {int(train_df['label'].sum())}, Neg: {int((train_df['label'] == 0).sum())}")
    print(f"Val set:   {len(val_df)} rows, Pos: {int(val_df['label'].sum())}, Neg: {int((val_df['label'] == 0).sum())}")
    print(f"Test set:  {len(test_df)} rows, Pos: {int(test_df['label'].sum())}, Neg: {int((test_df['label'] == 0).sum())}")

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved: {train_path}, {val_path}, {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for URL classifier.")
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    parser.add_argument("--subsample", type=int, default=50000, help="Total samples to extract for fast training")
    args = parser.parse_args()

    prepare_data(output_dir=args.output_dir, subsample_total=args.subsample)
