"""Verify Python Storage dtype dispatch and value round trips.

The original #673 static-link initialization failure is tested by the
standalone C++ executable in tests/storage_init_link_test.cpp. Importing
cytnx initializes the extension first, so Python cannot reproduce that
linking failure.
"""

import pytest

import cytnx
from cytnx import Type


def test_storage_default_dtype_dispatches_correctly():
    a = cytnx.Storage(10)
    assert a.size() == 10
    assert a.dtype() == Type.Double


@pytest.mark.parametrize(
    "dtype,value",
    [
        (Type.ComplexDouble, complex(1.5, -2.5)),
        (Type.ComplexFloat, complex(1.5, -2.5)),
        (Type.Double, -3.25),
        (Type.Float, -3.25),
        (Type.Int64, -12345),
        (Type.Uint64, 12345),
        (Type.Int32, -123),
        (Type.Uint32, 123),
        (Type.Int16, -12),
        (Type.Uint16, 12),
    ],
)
def test_storage_construction_every_dtype_dispatches_correctly(dtype, value):
    s = cytnx.Storage(3, dtype)
    assert s.size() == 3
    assert s.dtype() == dtype
    for i in range(3):
        s[i] = value
    assert [s[i] for i in range(3)] == [value, value, value]


def test_storage_construction_bool_dtype_round_trips_true_and_false():
    s = cytnx.Storage(2, Type.Bool)
    assert s.size() == 2
    assert s.dtype() == Type.Bool
    s[0] = False
    s[1] = True
    assert s[0] is False
    assert s[1] is True
