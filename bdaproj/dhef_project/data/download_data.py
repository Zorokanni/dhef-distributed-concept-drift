"""
============================================================================
D-HEF Project: Dataset Download / Synthetic Generation Script
============================================================================
Downloads the Credit Card Fraud dataset. If the download fails, generates a
synthetic dataset with matching schema (100k rows, 1% fraud rate).
============================================================================
"""

import os
import sys
import urllib.request
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_URL = (
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/creditcard.csv"
)
# Resolve paths relative to THIS file so the script works from any cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(SCRIPT_DIR, "creditcard.csv")

SEED = 42
N_ROWS = 100_000
N_FEATURES = 28
FRAUD_RATIO = 0.01  # 1 % fraud


def download_dataset() -> bool:
    """Attempt to download the real Credit Card Fraud dataset."""
    print(f"Attempting to download dataset from:\n  {DATASET_URL}")
    try:
        urllib.request.urlretrieve(DATASET_URL, SAVE_PATH)
        print("Download successful!")
        return True
    except Exception as exc:
        print(f"Download failed: {exc}")
        return False


def generate_synthetic_dataset() -> None:
    """
    Generate a synthetic credit-card-style dataset.
    - 28 features V1-V28 (standard normal)
    - Amount column (uniform 0-1000)
    - Time column (0 .. N_ROWS-1)
    - Class column: exactly 1 % fraud (Class=1)
    """
    print("Generating synthetic dataset ...")
    rng = np.random.RandomState(SEED)

    # --- Feature columns V1 .. V28 -------------------------------------------
    features = rng.randn(N_ROWS, N_FEATURES)
    col_names = [f"V{i}" for i in range(1, N_FEATURES + 1)]

    df = pd.DataFrame(features, columns=col_names)

    # --- Amount & Time --------------------------------------------------------
    df["Amount"] = rng.uniform(0, 1000, size=N_ROWS)
    df["Time"] = np.arange(N_ROWS, dtype=float)

    # --- Class label (1 % fraud) ----------------------------------------------
    n_fraud = int(N_ROWS * FRAUD_RATIO)  # 1 000
    labels = np.zeros(N_ROWS, dtype=int)
    fraud_indices = rng.choice(N_ROWS, size=n_fraud, replace=False)
    labels[fraud_indices] = 1
    df["Class"] = labels

    # --- Save -----------------------------------------------------------------
    df.to_csv(SAVE_PATH, index=False)
    print(f"Synthetic dataset saved ({N_ROWS} rows, {n_fraud} fraud).")


def print_class_distribution() -> None:
    """Load the saved CSV and print the class distribution."""
    df = pd.read_csv(SAVE_PATH)
    dist = df["Class"].value_counts().sort_index()
    print("\nClass distribution:")
    for cls, count in dist.items():
        pct = count / len(df) * 100
        print(f"  Class {cls}: {count:>7,} ({pct:.2f}%)")
    print(f"\nDataset ready at {SAVE_PATH}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    # Make sure the data/ directory exists
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # Try downloading; fall back to synthetic generation
    if not download_dataset():
        generate_synthetic_dataset()

    # Always print the distribution
    print_class_distribution()
