"""
============================================================================
D-HEF Project: Kafka Producer — Transaction Stream with Concept Drift
============================================================================
Reads creditcard.csv and streams records to the Kafka topic "transactions".
Injects two concept drifts:
  • Row 30 000 — swaps V1 and V2 (pattern change)
  • Row 60 000 — negates V1   (distribution shift)
============================================================================
"""

import os
import sys
import json
import time

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "data", "creditcard.csv")

KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC = "transactions"

SEND_DELAY = 0.01          # seconds between records
PROGRESS_INTERVAL = 1000   # print every N records
DRIFT_POINT_1 = 30_000     # swap V1 <-> V2
DRIFT_POINT_2 = 60_000     # negate V1


def create_producer():
    """Create and return a KafkaProducer with JSON serialisation."""
    from kafka import KafkaProducer

    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        print(f"Connected to Kafka at {KAFKA_BOOTSTRAP}")
        return producer
    except Exception as exc:
        print(
            f"ERROR: Could not connect to Kafka at {KAFKA_BOOTSTRAP}.\n"
            f"  Make sure Kafka is running (docker-compose up -d).\n"
            f"  Details: {exc}"
        )
        sys.exit(1)


def stream_transactions() -> None:
    """Read dataset and stream rows to Kafka, injecting concept drift."""
    # --- Load data ------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("  Run 'python data/download_data.py' first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records from {DATA_PATH}")

    # --- Create Kafka producer ------------------------------------------------
    producer = create_producer()

    total_sent = 0
    start_time = time.time()

    for idx, row in df.iterrows():
        record = row.to_dict()

        # ---- Concept Drift Injection -----------------------------------------
        if idx == DRIFT_POINT_1:
            # Drift 1: swap V1 and V2
            record["V1"], record["V2"] = record["V2"], record["V1"]
            print(f"\n*** DRIFT INJECTED AT ROW {DRIFT_POINT_1} (V1 <-> V2 swap) ***\n")

        elif idx >= DRIFT_POINT_1 and idx < DRIFT_POINT_2:
            # Continue the swap for all rows after drift point 1
            record["V1"], record["V2"] = record["V2"], record["V1"]

        if idx == DRIFT_POINT_2:
            print(f"\n*** DRIFT INJECTED AT ROW {DRIFT_POINT_2} (V1 negated) ***\n")

        if idx >= DRIFT_POINT_2:
            # Drift 2: negate V1
            record["V1"] = -record["V1"]

        # ---- Send to Kafka ---------------------------------------------------
        producer.send(KAFKA_TOPIC, value=record)
        total_sent += 1

        # ---- Progress --------------------------------------------------------
        if total_sent % PROGRESS_INTERVAL == 0:
            elapsed = time.time() - start_time
            rate = total_sent / elapsed if elapsed > 0 else 0
            print(
                f"Sent {total_sent} records... "
                f"({rate:.0f} rec/s)"
            )

        time.sleep(SEND_DELAY)

    # --- Flush & close --------------------------------------------------------
    producer.flush()
    producer.close()

    elapsed = time.time() - start_time
    print(f"\nProducer finished. Total records sent: {total_sent}")
    print(f"Elapsed time: {elapsed:.1f}s  ({total_sent / elapsed:.0f} rec/s)")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    stream_transactions()
