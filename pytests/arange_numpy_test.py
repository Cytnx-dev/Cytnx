"""cytnx.arange must match numpy.arange (#1083).

cytnx.arange follows numpy's ceil-based element count, so the two agree on the
element count AND the values -- including the floating-point edge cases where the
nominally half-open [start, end) range includes (or slightly exceeds) the
endpoint, e.g. np.arange(0.5, 0.8, 0.1) == [0.5, 0.6, 0.7, 0.8].
"""

import cytnx
import numpy as np
import pytest

# Ascending/descending, integer and fractional steps, the endpoint-inclusion FP
# cases, small scales, and single-element ranges.
CASES = [
    (0.0, 10.0, 1.0),
    (10.0, 40.0, 10.0),
    (0.0, 1.0, 0.25),
    (0.5, 0.8, 0.1),  # numpy includes the endpoint here
    (0.0, 1.0, 0.1),
    (0.0, 0.3, 0.1),
    (0.0, 0.4, 0.1),
    (0.1, 0.7, 0.11),
    (5.0, 0.0, -1.0),
    (10.0, 0.0, -2.0),
    (1.0, 0.0, -0.3),
    (0.0, 1e-15, 2e-15),  # small scale -> single element [0.0]
    (3.0, 3.5, 1.0),  # single element
]


@pytest.mark.parametrize("start,end,step", CASES)
def test_arange_matches_numpy(start, end, step):
    expected = np.arange(start, end, step, dtype=np.float64)
    got = cytnx.arange(start, end, step, cytnx.Type.Double).numpy()
    # Same element count (the endpoint-handling contract) ...
    assert got.shape == expected.shape
    # ... and same values.
    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize(
    "start,end,step",
    [
        (5.0, 5.0, 1.0),  # start == end
        (10.0, 0.0, 1.0),  # positive step, end < start
        (0.0, 10.0, -1.0),  # negative step, end > start
    ],
)
def test_arange_empty_matches_numpy(start, end, step):
    expected = np.arange(start, end, step, dtype=np.float64)
    got = cytnx.arange(start, end, step, cytnx.Type.Double).numpy()
    assert expected.shape == (0,)
    assert got.shape == (0,)


def test_arange_count_overload_matches_numpy():
    # arange(N) == np.arange(N)
    got = cytnx.arange(6).numpy()
    expected = np.arange(6, dtype=np.float64)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)
