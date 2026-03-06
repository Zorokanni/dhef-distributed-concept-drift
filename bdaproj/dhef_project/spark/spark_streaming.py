"""
============================================================================
D-HEF Project: Spark Structured Streaming — Main D-HEF Pipeline
============================================================================
Reads transactions from Kafka, distributes them across 4 partitions, runs
a minority-aware DriftDetector on each partition, and aggregates results
through MASAP to produce a synchronised global drift decision.

Results are appended to results/streaming_results.csv every micro-batch.
============================================================================
"""

import os
import sys
import json
import time
import datetime

import pandas as pd
import numpy as np

# Resolve project root so imports work regardless of cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from detector.drift_detector import DriftDetector
from detector.masap import MASAP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC = "transactions"
NUM_PARTITIONS = 4
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "streaming_results.csv")

# Feature columns expected in each JSON record
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time", "Class"]


def ensure_results_dir():
    """Create results/ directory if it does not exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ===========================================================================
# Global state — persisted across micro-batches
# ===========================================================================
# Detectors and MASAP are kept alive across batches so they accumulate state.
detectors = {pid: DriftDetector(partition_id=pid) for pid in range(NUM_PARTITIONS)}
masap = MASAP(n_partitions=NUM_PARTITIONS, drift_threshold=0.5)
batch_counter = 0
total_records_processed = 0
global_start_time = time.time()


def process_batch(batch_df, batch_id):
    """
    Called by foreachBatch on every micro-batch.

    Steps:
      1. Repartition data into NUM_PARTITIONS groups.
      2. Run each partition through its DriftDetector.
      3. Feed partition states to MASAP for a global drift decision.
      4. Log and persist results.
    """
    global batch_counter, total_records_processed

    if batch_df.rdd.isEmpty():
        return

    batch_counter += 1
    batch_start = time.time()

    # ---- Convert Spark DataFrame to Pandas for row-level processing ------
    pdf = batch_df.toPandas()
    n_rows = len(pdf)
    total_records_processed += n_rows

    print(f"\n{'='*70}")
    print(f"BATCH {batch_counter} | {n_rows} rows | Total: {total_records_processed}")
    print(f"{'='*70}")

    # ---- Split into partitions (round-robin) -----------------------------
    pdf["_partition"] = np.arange(len(pdf)) % NUM_PARTITIONS

    partition_states = []

    for pid in range(NUM_PARTITIONS):
        part_data = pdf[pdf["_partition"] == pid]

        for _, row in part_data.iterrows():
            v1 = float(row.get("V1", 0))
            label_value = row.get("Class", 0)
            if label_value is None or str(label_value) == "nan":
                label = 0
            else:
                label = int(label_value)
            drift = detectors[pid].update(v1, label)
            if drift:
                print(
                    f"  [Partition {pid}] Drift detected at record "
                    f"{total_records_processed}"
                )

        state = detectors[pid].get_state()
        partition_states.append(state)

        print(
            f"  Partition {pid}: minority={state['minority_seen']}, "
            f"drift_count={state['drift_count']}, "
            f"imbalance_ratio={state['imbalance_ratio']:.6f}"
        )

    # ---- MASAP global aggregation ----------------------------------------
    global_drift = masap.aggregate(partition_states)

    # ---- Compute batch-level metrics -------------------------------------
    batch_elapsed = time.time() - batch_start
    throughput = n_rows / batch_elapsed if batch_elapsed > 0 else 0

    # Simple per-batch minority F1 proxy: fraction of minority correctly
    # "covered" — in a real system this would use a classifier; here we
    # report the imbalance ratio as a proxy signal.
    avg_imbalance = np.mean([s["imbalance_ratio"] for s in partition_states])

    # ---- Persist to CSV --------------------------------------------------
    ensure_results_dir()
    row_data = {
        "batch_id": batch_counter,
        "timestamp": datetime.datetime.now().isoformat(),
        "records_in_batch": n_rows,
        "total_records": total_records_processed,
        "global_drift": int(global_drift),
        "throughput_rps": round(throughput, 2),
        "avg_imbalance_ratio": round(avg_imbalance, 6),
    }
    # Per-partition columns
    for s in partition_states:
        pid = s["partition_id"]
        row_data[f"p{pid}_minority"] = s["minority_seen"]
        row_data[f"p{pid}_drift_count"] = s["drift_count"]
        row_data[f"p{pid}_imbalance"] = s["imbalance_ratio"]

    results_df = pd.DataFrame([row_data])

    write_header = not os.path.exists(RESULTS_FILE)
    results_df.to_csv(RESULTS_FILE, mode="a", header=write_header, index=False)

    print(f"  Throughput: {throughput:.0f} rec/s | Global drift: {global_drift}")
    print(f"  Results appended to {RESULTS_FILE}")


# ===========================================================================
# Main — Spark Structured Streaming entry point
# ===========================================================================
def main():
    """Set up Spark, subscribe to Kafka topic, and start streaming."""
    try:
        import findspark
        findspark.init()
    except Exception:
        pass  # findspark optional if SPARK_HOME is already on PATH

    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, DoubleType, IntegerType
    )

    print("=" * 70)
    print("  D-HEF: Distributed Hybrid Ensemble Framework")
    print("  Spark Structured Streaming + MASAP")
    print("=" * 70)

    # ---- SparkSession -----------------------------------------------------
    # Try to create session WITH Kafka support; fall back to local-only mode
    # if Maven package resolution fails (network/environment issue)
    kafka_available = False
    try:
        spark = (
            SparkSession.builder
            .appName("DHEF")
            .master("local[4]")
            .config("spark.sql.shuffle.partitions", "4")
            .config(
                "spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"
            )
            .getOrCreate()
        )
        kafka_available = True
        print("  [OK] Spark session created WITH Kafka support")
    except Exception as kafka_error:
        print(
            f"  [WARNING] Could not load Kafka packages:\n"
            f"    {type(kafka_error).__name__}: {str(kafka_error)[:80]}"
        )
        # Fallback: create session WITHOUT external packages
        try:
            spark = (
                SparkSession.builder
                .appName("DHEF")
                .master("local[4]")
                .config("spark.sql.shuffle.partitions", "4")
                .getOrCreate()
            )
            print("  [OK] Spark session created (local mode, no Kafka)")
        except Exception as e:
            print(f"  [ERROR] Failed to create Spark session: {e}")
            sys.exit(1)
    
    spark.sparkContext.setLogLevel("WARN")

    # ---- Check Kafka availability -----------------------------------------
    if not kafka_available:
        print(
            f"\n{'='*70}\n"
            f"  KAFKA STREAMING NOT AVAILABLE\n"
            f"{'='*70}\n"
            f"  To use the full streaming pipeline with Kafka:\n"
            f"    1. Make sure Java/Spark can download dependencies\n"
            f"    2. Run: docker-compose up -d\n"
            f"    3. Run: python producer/kafka_producer.py\n"
            f"    4. Run: python spark/spark_streaming.py\n\n"
            f"  For now, use the simulation mode:\n"
            f"    • python baseline/naive_distribution.py\n"
            f"    • python experiments/run_experiments.py\n"
            f"{'='*70}\n"
        )
        spark.stop()
        sys.exit(0)  # Exit cleanly with success code

    # ---- Read stream from Kafka --------------------------------------------
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .load()
    )

    # ---- Define schema for JSON records ------------------------------------
    schema = StructType(
        [StructField(f"V{i}", DoubleType(), True) for i in range(1, 29)]
        + [
            StructField("Amount", DoubleType(), True),
            StructField("Time", DoubleType(), True),
            StructField("Class", IntegerType(), True),
        ]
    )

    # ---- Parse JSON payload ------------------------------------------------
    parsed_stream = (
        raw_stream
        .selectExpr("CAST(value AS STRING) as json_str")
        .select(F.from_json(F.col("json_str"), schema).alias("data"))
        .select("data.*")
    )

    # ---- Start foreachBatch streaming query ---------------------------------
    checkpoint_dir = os.path.join(PROJECT_DIR, "checkpoint")

    query = (
        parsed_stream.writeStream
        .foreachBatch(process_batch)
        .option("checkpointLocation", checkpoint_dir)
        .trigger(processingTime="10 seconds")
        .start()
    )

    print(f"\nStreaming started — listening on {KAFKA_TOPIC} ...")
    print("Press Ctrl+C to stop.\n")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\nStopping stream ...")
        query.stop()
        spark.stop()
        elapsed = time.time() - global_start_time
        print(
            f"Done. Processed {total_records_processed} records in "
            f"{elapsed:.1f}s ({total_records_processed / elapsed:.0f} rec/s)"
        )


if __name__ == "__main__":
    main()
