"""
============================================================================
D-HEF Project: Drift Detector Module
============================================================================
Provides two drift detector classes built on the ADWIN algorithm:

  • DriftDetector       – minority-aware; tracks class distribution per
                          partition so MASAP can weight votes.
  • NaiveDriftDetector  – baseline; no minority/majority tracking, just
                          overall error-rate ADWIN.

Uses a pure-Python ADWIN implementation (detector/adwin.py) to avoid
the 'river' library dependency which requires Cython compilation.
============================================================================
"""

# Use our pure-Python ADWIN implementation (no C compilation needed)
try:
    from detector.adwin import ADWIN
except ImportError:
    # Fallback for running this file directly (e.g. self-test)
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from adwin import ADWIN


class DriftDetector:
    """
    Partition-level drift detector that is *minority-aware*.

    It wraps the ADWIN change detector and additionally counts how many
    minority vs. majority samples this partition has seen, enabling MASAP
    to give more weight to partitions that observe more minority-class data.
    """

    def __init__(self, partition_id: int, minority_class: int = 1):
        """
        Parameters
        ----------
        partition_id   : identifier for the Spark partition this detector serves
        minority_class : label value that represents the minority class (default 1)
        """
        self.partition_id = partition_id
        self.minority_class = minority_class

        # ADWIN detector from the river library
        self.detector = ADWIN()

        # Counters
        self.minority_seen = 0
        self.majority_seen = 0
        self.drift_count = 0

    # --------------------------------------------------------------------- #
    def update(self, value: float, true_label: int) -> bool:
        """
        Feed a new observation to the detector.

        Parameters
        ----------
        value      : numeric value to monitor (e.g. feature V1)
        true_label : ground-truth class label for this sample

        Returns
        -------
        True if ADWIN detected a change (drift) on this update.
        """
        # Track minority / majority counts
        if true_label == self.minority_class:
            self.minority_seen += 1
        else:
            self.majority_seen += 1

        # Feed value to ADWIN
        self.detector.update(value)
        drift_detected = self.detector.drift_detected

        if drift_detected:
            self.drift_count += 1

        return drift_detected

    # --------------------------------------------------------------------- #
    def get_state(self) -> dict:
        """Return a summary dict describing this detector's current state."""
        total = self.minority_seen + self.majority_seen
        imbalance_ratio = (self.minority_seen / total) if total > 0 else 0.0

        return {
            "partition_id": self.partition_id,
            "minority_seen": self.minority_seen,
            "majority_seen": self.majority_seen,
            "drift_count": self.drift_count,
            "imbalance_ratio": round(imbalance_ratio, 6),
            "detector_state": "active",
        }


# ========================================================================= #
# Naive Baseline Detector (no minority/majority awareness)
# ========================================================================= #

class NaiveDriftDetector:
    """
    Baseline drift detector that treats all samples equally.

    It does NOT track minority vs. majority classes — it simply runs ADWIN
    on the overall observed values.  This is used in the naive baseline to
    demonstrate partition desynchronisation.
    """

    def __init__(self, partition_id: int, minority_class: int = 1):
        self.partition_id = partition_id
        self.minority_class = minority_class

        self.detector = ADWIN()
        self.total_seen = 0
        self.drift_count = 0

    # --------------------------------------------------------------------- #
    def update(self, value: float, true_label: int) -> bool:
        """Feed a value; label is accepted for API compatibility but unused."""
        self.total_seen += 1

        self.detector.update(value)
        drift_detected = self.detector.drift_detected

        if drift_detected:
            self.drift_count += 1

        return drift_detected

    # --------------------------------------------------------------------- #
    def get_state(self) -> dict:
        return {
            "partition_id": self.partition_id,
            "minority_seen": 0,       # not tracked
            "majority_seen": 0,       # not tracked
            "drift_count": self.drift_count,
            "imbalance_ratio": 0.0,   # not applicable
            "detector_state": "active",
        }


# ===========================================================================
# Quick self-test
# ===========================================================================
if __name__ == "__main__":
    import numpy as np

    rng = np.random.RandomState(42)

    print("=== DriftDetector self-test ===")
    det = DriftDetector(partition_id=0)
    for i in range(5000):
        val = rng.normal(0, 1) if i < 2500 else rng.normal(3, 1)
        label = 1 if rng.rand() < 0.01 else 0
        if det.update(val, label):
            print(f"  Drift detected at sample {i}")
    print(f"  Final state: {det.get_state()}")

    print("\n=== NaiveDriftDetector self-test ===")
    ndet = NaiveDriftDetector(partition_id=0)
    for i in range(5000):
        val = rng.normal(0, 1) if i < 2500 else rng.normal(3, 1)
        label = 1 if rng.rand() < 0.01 else 0
        if ndet.update(val, label):
            print(f"  Drift detected at sample {i}")
    print(f"  Final state: {ndet.get_state()}")
