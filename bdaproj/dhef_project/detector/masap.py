"""
============================================================================
D-HEF Project: MASAP — Minority-Aware Synchronized Aggregation Protocol
============================================================================
MASAP is the core research contribution of D-HEF.  It aggregates per-
partition drift votes using *minority-class-weighted voting* so that
partitions that have observed more minority-class samples (and are thus
better informed about the rare class) exert a proportionally larger
influence on the global drift decision.

Without MASAP, partitions with very few minority samples may detect drift
much later (or not at all), causing "desynchronisation".
============================================================================
"""

from typing import Dict, List


class MASAP:
    """
    Minority-Aware Synchronized Aggregation Protocol.

    Parameters
    ----------
    n_partitions    : number of Spark partitions being monitored
    drift_threshold : weighted-vote ratio at which global drift is confirmed
    """

    def __init__(self, n_partitions: int = 4, drift_threshold: float = 0.5):
        self.n_partitions = n_partitions
        self.drift_threshold = drift_threshold

        # Track each partition's last known drift_count so we can detect
        # *new* drifts between successive aggregate() calls.
        self.last_drift_counts: Dict[int, int] = {
            pid: 0 for pid in range(n_partitions)
        }

        # Most recent sync report (populated by aggregate)
        self._last_report: dict = {}

    # --------------------------------------------------------------------- #
    def aggregate(self, partition_states: List[dict]) -> bool:
        """
        Aggregate per-partition drift detector states and decide whether
        a *global* drift has occurred.

        Steps
        -----
        1. Each partition casts a binary vote: 1 if its drift_count has
           increased since the last call, 0 otherwise.
        2. Each vote is weighted by that partition's `minority_seen` count.
        3. If the weighted vote ratio ≥ drift_threshold → global drift.

        Parameters
        ----------
        partition_states : list of dicts produced by DriftDetector.get_state()

        Returns
        -------
        True if global drift is confirmed; False otherwise.
        """
        votes = []
        weights = []
        details = []

        for state in partition_states:
            pid = state["partition_id"]
            current_dc = state["drift_count"]
            prev_dc = self.last_drift_counts.get(pid, 0)

            # Binary vote: did this partition detect NEW drift?
            vote = 1 if current_dc > prev_dc else 0

            # Weight = number of minority samples this partition has seen
            # (minimum of 1 to avoid zero-weight partitions entirely)
            weight = max(state["minority_seen"], 1)

            votes.append(vote)
            weights.append(weight)
            details.append(
                {
                    "partition_id": pid,
                    "vote": vote,
                    "weight": weight,
                    "drift_count": current_dc,
                    "minority_seen": state["minority_seen"],
                }
            )

            # Update stored drift count
            self.last_drift_counts[pid] = current_dc

        # ---- Compute weighted vote ratio ------------------------------------
        total_weight = sum(weights)
        weighted_yes = sum(v * w for v, w in zip(votes, weights))
        weighted_ratio = weighted_yes / total_weight if total_weight > 0 else 0.0

        global_drift = weighted_ratio >= self.drift_threshold

        # ---- Store report ----------------------------------------------------
        self._last_report = {
            "partition_details": details,
            "weighted_vote_ratio": round(weighted_ratio, 4),
            "drift_threshold": self.drift_threshold,
            "global_drift": global_drift,
        }

        # ---- Console feedback ------------------------------------------------
        if global_drift:
            print(
                f"MASAP: Global drift CONFIRMED  "
                f"(weighted ratio = {weighted_ratio:.4f} >= {self.drift_threshold})"
            )
        else:
            print(
                f"MASAP: No global drift  "
                f"(weighted ratio = {weighted_ratio:.4f} < {self.drift_threshold})"
            )

        return global_drift

    # --------------------------------------------------------------------- #
    def get_sync_report(self) -> dict:
        """
        Return a detailed report of the most recent aggregation round.

        Keys: partition_details, weighted_vote_ratio, drift_threshold,
              global_drift
        """
        return self._last_report


# ===========================================================================
# Quick self-test
# ===========================================================================
if __name__ == "__main__":
    print("=== MASAP self-test ===\n")

    masap = MASAP(n_partitions=4, drift_threshold=0.5)

    # Simulate: partition 0 and 1 see drift; partition 2 and 3 do not
    states_round_1 = [
        {"partition_id": 0, "minority_seen": 50, "majority_seen": 4950,
         "drift_count": 1, "imbalance_ratio": 0.01, "detector_state": "active"},
        {"partition_id": 1, "minority_seen": 40, "majority_seen": 4960,
         "drift_count": 1, "imbalance_ratio": 0.008, "detector_state": "active"},
        {"partition_id": 2, "minority_seen": 5, "majority_seen": 4995,
         "drift_count": 0, "imbalance_ratio": 0.001, "detector_state": "active"},
        {"partition_id": 3, "minority_seen": 3, "majority_seen": 4997,
         "drift_count": 0, "imbalance_ratio": 0.0006, "detector_state": "active"},
    ]

    result = masap.aggregate(states_round_1)
    print(f"  Global drift? {result}")
    print(f"  Report: {masap.get_sync_report()}\n")

    # Round 2: all partitions detect drift
    states_round_2 = [
        {"partition_id": 0, "minority_seen": 100, "majority_seen": 9900,
         "drift_count": 2, "imbalance_ratio": 0.01, "detector_state": "active"},
        {"partition_id": 1, "minority_seen": 80, "majority_seen": 9920,
         "drift_count": 2, "imbalance_ratio": 0.008, "detector_state": "active"},
        {"partition_id": 2, "minority_seen": 10, "majority_seen": 9990,
         "drift_count": 1, "imbalance_ratio": 0.001, "detector_state": "active"},
        {"partition_id": 3, "minority_seen": 6, "majority_seen": 9994,
         "drift_count": 1, "imbalance_ratio": 0.0006, "detector_state": "active"},
    ]

    result = masap.aggregate(states_round_2)
    print(f"  Global drift? {result}")
    print(f"  Report: {masap.get_sync_report()}")
