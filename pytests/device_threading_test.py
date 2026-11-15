"""Check the Device API removal and HPTT environment handling."""

import numpy as np
import pytest

import cytnx


def test_device_describes_devices() -> None:
    """Keep device discovery while removing the obsolete thread-count copy."""
    assert not hasattr(cytnx.Device, "Ncpus")
    assert cytnx.Device.cpu == -1
    assert cytnx.Device.cuda == 0
    assert isinstance(cytnx.Device.Ngpus, int)
    assert "CPU" in cytnx.Device.getname(cytnx.Device.cpu)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64,
                                  np.complex128])
@pytest.mark.parametrize("inplace", [False, True])
def test_transpose_with_hptt_override(dtype, inplace, monkeypatch) -> None:
    """Preserve all coefficients and dtypes as the override changes."""
    values = (np.arange(315).reshape(5, 7, 9) - 157) * 0.25
    if np.issubdtype(dtype, np.complexfloating):
        values = values + 1j * (values + 0.5)
    source = values.astype(dtype)
    expected = source.transpose(2, 0, 1)
    for count in ("1", "3", "1"):
        monkeypatch.setenv("HPTT_NUM_THREADS", count)
        tensor = cytnx.from_numpy(source.copy()).permute(2, 0, 1)
        if inplace:
            tensor.contiguous_()
        else:
            tensor = tensor.contiguous()
        actual = tensor.numpy()
        assert actual.dtype == source.dtype
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif("UNI_HPTT" not in cytnx.__variant_info__,
                    reason="HPTT is disabled in this build")
@pytest.mark.parametrize("value", ["", "0", "-1", "2,1", "invalid"])
def test_invalid_hptt_override_raises(value, monkeypatch) -> None:
    """Translate HPTT configuration errors without corrupting the source."""
    source = cytnx.arange(315).reshape(5, 7, 9)
    monkeypatch.setenv("HPTT_NUM_THREADS", value)
    with pytest.raises(cytnx.CytnxError, match="HPTT_NUM_THREADS"):
        source.permute(2, 0, 1).contiguous()
    monkeypatch.setenv("HPTT_NUM_THREADS", "1")
    actual = source.permute(2, 0, 1).contiguous().numpy()
    np.testing.assert_array_equal(actual,
                                  np.arange(315).reshape(5, 7, 9).transpose(2, 0, 1))
