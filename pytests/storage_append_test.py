"""Regression tests for #941 ruling 2: capacity_ removed, append() = exact
realloc (documented O(n) per call, numpy-like).

Storage.capacity() is gone from the Python API; append() keeps working with
the only observable contract being size growth + element preservation.
"""

import pytest

import cytnx


def test_storage_append_grows_and_preserves():
    s = cytnx.Storage(3, cytnx.Type.Double)
    for i in range(3):
        s[i] = float(i + 1)
    s.append(4.0)
    assert s.size() == 4
    assert [s[i] for i in range(4)] == [1.0, 2.0, 3.0, 4.0]
    s.append(5.0)
    assert s.size() == 5
    assert s[4] == 5.0


def test_storage_capacity_removed():
    s = cytnx.Storage(3, cytnx.Type.Double)
    assert not hasattr(s, "capacity"), (
        "Storage.capacity() was removed (#941 ruling 2): storage always "
        "allocates exactly size() elements")


def test_storage_resize_exact():
    s = cytnx.Storage(4, cytnx.Type.Int64)
    for i in range(4):
        s[i] = i + 10
    s.resize(2)
    assert s.size() == 2
    assert [s[0], s[1]] == [10, 11]
    s.resize(5)
    assert s.size() == 5
    assert [s[0], s[1]] == [10, 11]
