"""Unit tests for the candidate-sensitivity statistics (synthetic arrays only)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate_sensitivity as C  # noqa: E402


def test_identical_maps_have_zero_distance_and_unit_correlation():
    m = np.arange(49, dtype=float).reshape(7, 7) - 10
    s = C.pair_stats(m, m.copy())
    assert s["l1_distance"] == 0.0
    assert abs(s["correlation"] - 1.0) < 1e-12
    assert not s["degenerate_pos"] and not s["degenerate_neg"]


def test_constant_map_correlation_is_undefined_not_one():
    m = np.ones((7, 7))
    s = C.pair_stats(m, m.copy())
    assert math.isnan(s["correlation"])
    assert s["degenerate_pos"] and s["degenerate_neg"]


def test_zero_positive_part_gives_undefined_distance():
    s = C.pair_stats(-np.ones((7, 7)) - np.eye(7), np.eye(7))
    assert math.isnan(s["l1_distance"])


def test_disjoint_supports_have_maximal_distance():
    a = np.zeros((7, 7)); a[0, 0] = 1
    b = np.zeros((7, 7)); b[6, 6] = 1
    assert abs(C.pair_stats(a, b)["l1_distance"] - 2.0) < 1e-12


def test_nonfinite_map_flagged_degenerate():
    a = np.zeros((7, 7)); a[0, 0] = np.nan
    s = C.pair_stats(a, np.eye(7))
    assert s["degenerate_pos"] and math.isnan(s["raw_max_abs_pos"])


def test_centre_prior_peaks_at_centre():
    g = C.centre_prior((7, 7)).numpy()
    assert g.argmax() == 24 and g.min() > 0


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); n += 1
    print(f"{n} candidate-sensitivity tests passed")
