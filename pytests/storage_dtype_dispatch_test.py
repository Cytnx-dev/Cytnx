"""Verify Python Storage dtype dispatch and value round trips.

The original #673 static-link initialization failure is tested by the
standalone C++ executable in tests/storage_init_link_test.cpp. Importing
cytnx initializes the extension first, so Python cannot reproduce that
linking failure.
"""

import struct

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


def test_storage_construction_void_dtype_raises():
    # Type.Void passes the dtype range check, and before #1073 it called a null
    # function pointer.
    with pytest.raises(cytnx.CytnxError):
        cytnx.Storage(1, Type.Void)


def _write_storage_file(path, dtype):
    """Writes a one-element Storage file in the layout Storage.Load reads.

    The header is written by hand so the dtype field can hold values that
    Storage.Save never produces.

    Args:
        path: Destination file path.
        dtype: Value for the dtype field of the header.
    """
    magic, size, device_cpu = 999, 1, -1
    header = struct.pack("=IQIi", magic, size, int(dtype), device_cpu)
    path.write_bytes(header + struct.pack("=d", 1.5))


def test_storage_load_reads_hand_written_file(tmp_path):
    path = tmp_path / "valid_dtype.bin"
    _write_storage_file(path, Type.Double)
    loaded = cytnx.Storage.Load(str(path))
    assert loaded.dtype() == Type.Double
    assert loaded.size() == 1
    assert loaded[0] == 1.5


@pytest.mark.parametrize("dtype", [Type.Void, 99])
def test_storage_load_rejects_invalid_dtype_in_file(tmp_path, dtype):
    path = tmp_path / "invalid_dtype.bin"
    _write_storage_file(path, dtype)
    with pytest.raises(cytnx.CytnxError):
        cytnx.Storage.Load(str(path))
