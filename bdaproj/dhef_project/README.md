# D-HEF: Distributed Hybrid Ensemble Framework

**Real-time concept drift detection in imbalanced data streams using Apache Spark Structured Streaming + Kafka.**

---

## Research Contribution

When drift detectors (ADWIN) are naively distributed across Spark partitions, partitions that receive fewer minority-class samples detect concept drift significantly later than those with more — we call this **partition desynchronisation**. In the Credit Card Fraud dataset (1 % fraud rate), hash-based partitioning can cause a gap of 500+ rows between the first and last partition to detect the same drift event. D-HEF solves this with **MASAP (Minority-Aware Synchronized Aggregation Protocol)**, which weights each partition's drift vote by how many minority samples it has observed, ensuring the global drift decision is driven by the most-informed partitions.

---

## Architecture

```
┌─────────────┐      ┌──────────┐      ┌──────────────────────────┐
│  CSV Data   │─────▶│  Kafka   │─────▶│  Spark Structured        │
│  Producer   │      │  Topic   │      │  Streaming (4 partitions)│
└─────────────┘      └──────────┘      └────────────┬─────────────┘
                                                    │
                                       ┌────────────▼─────────────┐
                                       │  Per-Partition ADWIN     │
                                       │  DriftDetectors          │
                                       └────────────┬─────────────┘
                                                    │
                                       ┌────────────▼─────────────┐
                                       │  MASAP Aggregator        │
                                       │  (weighted voting)       │
                                       └────────────┬─────────────┘
                                                    │
                                       ┌────────────▼─────────────┐
                                       │  Results CSV + Dashboard │
                                       └──────────────────────────┘
```

---

## Prerequisites

- **Python 3.9+**
- **Docker Desktop** (for Kafka + Zookeeper)
- **Java 8/11** (required by PySpark)

---

## Setup Instructions

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Kafka & Zookeeper

```bash
docker-compose up -d
```

Wait ~10 seconds for Kafka to become healthy.

### 3. Download / generate dataset

```bash
python data/download_data.py
```

### 4. Start the Kafka producer (Terminal 1)

```bash
python producer/kafka_producer.py
```

This streams records to the `transactions` topic and injects concept drift at rows 30 000 and 60 000.

### 5. Start the D-HEF streaming pipeline (Terminal 2)

```bash
python spark/spark_streaming.py
```

### 6. Launch the dashboard (Terminal 3)

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) to see live metrics.

---

## Running the Naive Baseline

To demonstrate the desynchronisation problem **without** MASAP:

```bash
python baseline/naive_distribution.py
```

This runs a simulation over the dataset and prints the desynchronisation gap.

---

## Running Experiments

After both `streaming_results.csv` and `naive_results.csv` exist in `results/`:

```bash
python experiments/run_experiments.py
```

This produces:

| File | Description |
|------|-------------|
| `results/desync_gap.png` | **Key figure** — per-partition drift timing, Naive vs D-HEF |
| `results/minority_f1.png` | Minority class F1 comparison |
| `results/throughput.png` | Processing throughput over time |

---

## Expected Results

### `desync_gap.png` (Key Figure)

The **left panel** (Naive) should show partition drift detection curves that are spread apart — Partition 0 detects drift earliest while Partition 3 detects it much later (gap of ~500+ rows). The **right panel** (D-HEF/MASAP) should show all four partition curves aligned closely, demonstrating that MASAP successfully synchronises drift detection across partitions regardless of minority-class distribution.

### Desynchronisation Gap

| | Naive | D-HEF |
|---|---|---|
| Gap (rows) | ~500–2 500 | ~0 |

---

## Project Structure

```
dhef_project/
├── docker-compose.yml       # Kafka + Zookeeper
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── data/
│   └── download_data.py     # Dataset download / generation
├── producer/
│   └── kafka_producer.py    # Kafka producer with drift injection
├── detector/
│   ├── drift_detector.py    # ADWIN-based drift detectors
│   └── masap.py             # MASAP aggregation protocol
├── spark/
│   └── spark_streaming.py   # Main Spark Structured Streaming pipeline
├── baseline/
│   └── naive_distribution.py# Naive baseline (proves desynchronisation)
├── experiments/
│   └── run_experiments.py   # Metrics computation & plot generation
├── dashboard/
│   └── app.py               # Streamlit real-time dashboard
└── results/                 # Output CSVs and plots
```

---

## License

Academic research project — BDA course, Semester 2.
