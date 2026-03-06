"""
============================================================================
D-HEF Project: Pure-Python ADWIN (ADaptive WINdowing) Implementation
============================================================================
A pure-Python implementation of the ADWIN change detector algorithm,
removing the dependency on the 'river' library (which requires Cython
compilation and may not install on newer Python versions).

Reference:
  Bifet & Gavalda (2007). "Learning from Time-Changing Data with Adaptive
  Windowing." Proc. 7th SIAM Int'l Conf. on Data Mining.

The algorithm maintains a variable-length window of recent items and
detects distribution change by comparing the means of two sub-windows.
When a statistically significant difference is found, the older portion
is dropped (= drift detected).
============================================================================
"""

import math


class ADWIN:
    """
    ADaptive WINdowing algorithm for concept drift detection.

    Parameters
    ----------
    delta : float, default=0.002
        Confidence parameter.  Smaller values make the detector less
        sensitive (fewer false positives, but slower to react).
    """

    def __init__(self, delta: float = 0.0001):
        self.delta = delta
        self._window = []           # list of observed values
        self._total = 0.0           # running sum
        self._variance = 0.0        # running M2 (for Welford's)
        self._count = 0             # number of elements
        self.drift_detected = False # flag set after each update()

    # ------------------------------------------------------------------ #
    def update(self, value: float) -> None:
        """
        Add a new observation and check for drift.

        After calling this method, inspect ``self.drift_detected`` to see
        whether a distribution change was flagged.
        """
        self._window.append(value)
        self._count += 1
        self._total += value

        self.drift_detected = False

        if self._count < 10:
            # Not enough data to test yet
            return

        # --- Try splitting the window at every power-of-2 boundary ----------
        # (heuristic to keep it O(log n) per update on average)
        found_cut = False
        n = len(self._window)

        step = max(1, n // 10)  # check ~10 split points for efficiency
        for i in range(step, n - step, step):
            # Sub-window 0: oldest  [0 .. i-1]
            # Sub-window 1: newest  [i .. n-1]
            n0 = i
            n1 = n - i
            mean0 = sum(self._window[:i]) / n0
            mean1 = sum(self._window[i:]) / n1

            # Hoeffding-style bound
            m = 1.0 / n0 + 1.0 / n1
            eps_cut = math.sqrt(
                0.5 * m * math.log(4.0 / self.delta)
            )

            if abs(mean0 - mean1) >= eps_cut:
                # Significant change found — drop the old sub-window
                self._window = self._window[i:]
                self._total = sum(self._window)
                self._count = len(self._window)
                self.drift_detected = True
                found_cut = True
                break

        # Prevent unbounded memory growth (cap at 5000)
        if not found_cut and len(self._window) > 5000:
            trim = len(self._window) - 5000
            self._window = self._window[trim:]
            self._total = sum(self._window)
            self._count = len(self._window)

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return (
            f"ADWIN(delta={self.delta}, window_size={self._count}, "
            f"drift_detected={self.drift_detected})"
        )


# ===========================================================================
# Quick self-test
# ===========================================================================
if __name__ == "__main__":
    import random

    random.seed(42)
    detector = ADWIN(delta=0.0001)

    print("=== ADWIN self-test ===")
    print("Feeding 2500 samples from N(0,1), then 2500 from N(3,1).\n")

    for i in range(5000):
        val = random.gauss(0, 1) if i < 2500 else random.gauss(3, 1)
        detector.update(val)
        if detector.drift_detected:
            print(f"  Drift detected at sample {i}")

    print(f"\nFinal state: {detector}")
    print("Done.")
