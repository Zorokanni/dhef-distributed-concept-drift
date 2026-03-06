"""
============================================================================
D-HEF Project: Naive Baseline — Demonstrates Partition Desynchronisation
============================================================================
This file proves the PROBLEM that D-HEF solves:

When drift detectors are naively distributed across Spark partitions without
minority-class awareness, partitions that receive fewer minority samples
detect drift significantly later than partitions with more minority samples.

The script can operate in two modes:
  1. LIVE MODE  — reads from Kafka  (same as spark_streaming.py but naive)
  2. SIMULATION — reads directly from CSV with artificially skewed minority
                  distribution to clearly demonstrate desynchronisation

By default it runs in SIMULATION mode for easy demonstration.
============================================================================
"""

import os
import sys
import time
import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from detector.drift_detector import NaiveDriftDetector

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(PROJECT_DIR, "data", "creditcard.csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "naive_results.csv")
NUM_PARTITIONS = 4
DRIFT_POINT_1 = 30_000   # row at which concept drift 1 is injected
DRIFT_POINT_2 = 60_000   # row at which concept drift 2 is injected
SEED = 42
BATCH_SIZE = 5_000        # rows per simulated micro-batch


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ===========================================================================
# Simulation mode  (deterministic, clearly shows desynchronisation)
# ===========================================================================
def run_simulation():
    """
    Simulate the naive-distribution scenario.

    Key idea: we deliberately skew the minority-class distribution so that
      • Partition 0 receives ~40 % of all minority (fraud) samples
      • Partition 3 receives ~5 %
    This mirrors real-world hash-based partitioning where the rare class
    is not uniformly distributed.

    We then inject a concept drift at DRIFT_POINT_1 (shift in V1 by +3 σ)
    and watch how long it takes each partition's NaiveDriftDetector to fire.
    """
    rng = np.random.RandomState(SEED)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("  Run 'python data/download_data.py' first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    n_rows = min(len(df), 80_000)  # cap for speed
    df = df.iloc[:n_rows].copy()

    print("=" * 70)
    print("  NAIVE BASELINE — Partition Desynchronisation Demonstration")
    print("=" * 70)

    # ---- Assign partitions with SKEWED minority distribution ----------------
    # We want partition 0 to get the most minority samples, partition 3 the
    # fewest.  For majority-class rows we use round-robin.
    minority_mask = df["Class"] == 1
    majority_mask = ~minority_mask

    minority_idx = df.index[minority_mask].tolist()
    majority_idx = df.index[majority_mask].tolist()

    # Skewed allocation weights: [40%, 30%, 20%, 10%]
    minority_weights = [0.40, 0.30, 0.20, 0.10]
    partition_col = np.zeros(n_rows, dtype=int)

    # Distribute minority samples according to weights
    rng.shuffle(minority_idx)
    cursor = 0
    for pid, weight in enumerate(minority_weights):
        count = int(len(minority_idx) * weight)
        end = min(cursor + count, len(minority_idx))
        for i in minority_idx[cursor:end]:
            partition_col[i] = pid
        cursor = end
    # Any remaining minority samples go to partition 0
    for i in minority_idx[cursor:]:
        partition_col[i] = 0

    # Majority: round-robin
    for rank, i in enumerate(majority_idx):
        partition_col[i] = rank % NUM_PARTITIONS

    df["_partition"] = partition_col

    # ---- Inject concept drift into the data --------------------------------
    # Drift 1 at DRIFT_POINT_1: shift V1 by +3
    df.loc[DRIFT_POINT_1:DRIFT_POINT_2 - 1, "V1"] += 3.0
    # Drift 2 at DRIFT_POINT_2: negate V1
    df.loc[DRIFT_POINT_2:, "V1"] *= -1

    # ---- Run naive detectors -----------------------------------------------
    detectors = {pid: NaiveDriftDetector(partition_id=pid) for pid in range(NUM_PARTITIONS)}
    first_drift_row = {}   # partition_id -> row index of first drift detection
    all_results = []
    batch_counter = 0

    for batch_start in range(0, n_rows, BATCH_SIZE):
        batch_counter += 1
        batch_end = min(batch_start + BATCH_SIZE, n_rows)
        batch = df.iloc[batch_start:batch_end]

        partition_states = []

        for pid in range(NUM_PARTITIONS):
            part_data = batch[batch["_partition"] == pid]

            for _, row in part_data.iterrows():
                v1 = float(row["V1"])
                label = int(row["Class"])
                drift = detectors[pid].update(v1, label)

                if drift and pid not in first_drift_row:
                    first_drift_row[pid] = row.name  # index = approximate row #
                    print(
                        f"  ** Partition {pid} detected drift at row ~{row.name} **"
                    )

            state = detectors[pid].get_state()
            partition_states.append(state)

        # ---- Log batch summary -----------------------------------------------
        row_data = {
            "batch_id": batch_counter,
            "timestamp": datetime.datetime.now().isoformat(),
            "batch_start_row": batch_start,
            "batch_end_row": batch_end,
        }
        for s in partition_states:
            pid = s["partition_id"]
            row_data[f"p{pid}_drift_count"] = s["drift_count"]
            row_data[f"p{pid}_total_seen"] = s.get("total_seen", detectors[pid].total_seen)

        all_results.append(row_data)

        # Show per-partition drift counts
        counts = ", ".join(
            f"P{s['partition_id']}={s['drift_count']}" for s in partition_states
        )
        print(
            f"  Batch {batch_counter} (rows {batch_start}-{batch_end}): "
            f"drift counts [{counts}]"
        )

    # ---- Desynchronisation analysis -----------------------------------------
    print("\n" + "=" * 70)
    print("  DESYNCHRONISATION ANALYSIS")
    print("=" * 70)

    if first_drift_row:
        rows_sorted = sorted(first_drift_row.items(), key=lambda x: x[1])
        earliest_pid, earliest_row = rows_sorted[0]
        latest_pid, latest_row = rows_sorted[-1]
        desync_gap = latest_row - earliest_row

        print(f"\n  First partition to detect drift : Partition {earliest_pid} at row {earliest_row}")
        print(f"  Last  partition to detect drift : Partition {latest_pid}  at row {latest_row}")
        print(f"  Desynchronization Gap: {desync_gap} rows")
        print()

        for pid, row_num in rows_sorted:
            delay = row_num - earliest_row
            print(f"    Partition {pid}: drift at row {row_num}  (delay = {delay} rows)")
    else:
        print("  No drift detected by any partition.")
        desync_gap = 0

    # ---- Save results --------------------------------------------------------
    ensure_results_dir()
    results_df = pd.DataFrame(all_results)
    results_df["desync_gap"] = desync_gap if first_drift_row else 0
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\n  Results saved to {RESULTS_FILE}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    run_simulation()
