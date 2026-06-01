import os
import struct
import tempfile

import numpy as np
import pytest

import cytnx


# ---- Bond.Load_ stale-metadata reset ----

def test_bond_load_resets_stale_metadata(tmp_path):
    """In-place Load_ must not inherit stale qnums/degs/syms from a previously Bond."""
    fpath = str(tmp_path / "bond_plain.cytnx")

    # save a plain non-symmetric Bond
    bd_plain = cytnx.Bond(5)
    bd_plain.Save(fpath)

    # build a symmetric Bond, then Load_ the plain payload into it
    bd_sym = cytnx.Bond(cytnx.BD_KET, [[0], [1], [2]], [3, 3, 3])
    assert bd_sym.Nsym() != 0
    assert len(bd_sym.qnums()) != 0

    bd_sym.Load_(fpath)
    assert bd_sym == bd_plain, "Load_ should fully reconstruct the saved Bond"
    assert bd_sym.dim() == 5
    assert bd_sym.Nsym() == 0, "stale syms leaked into a non-symmetric Bond"
    assert len(bd_sym.qnums()) == 0, "stale qnums leaked into a non-symmetric Bond"


# ---- UniTensor.Load_ stale-name reset ----

def test_unitensor_load_resets_stale_name(tmp_path):
    """In-place Load_ must not inherit a stale name from a previously named UniTensor when the
    on-disk payload has an empty name."""
    fpath = str(tmp_path / "ut_anon.cytnx")

    bonds = [cytnx.Bond(3), cytnx.Bond(2)]
    labels = ["a", "b"]

    # save a UniTensor with empty name
    ut_anon = cytnx.UniTensor(bonds, labels, 1)
    assert ut_anon.name() == ""
    ut_anon.Save(fpath)

    # build a named UniTensor and Load_ the anonymous payload into it
    ut_named = cytnx.UniTensor(bonds, labels, 1)
    ut_named.set_name("stale_name")
    assert ut_named.name() == "stale_name"
    ut_named.Load_(fpath)
    assert ut_named.name() == "", "stale UniTensor name leaked when loading an empty-name payload"
    # full equivalence with the saved object (the bindings don't expose __eq__ for UniTensor,
    # so compare the user-visible attributes explicitly).
    assert ut_named.name() == ut_anon.name()
    assert ut_named.labels() == ut_anon.labels()
    assert ut_named.rowrank() == ut_anon.rowrank()
    assert ut_named.dtype() == ut_anon.dtype()
    for a, b in zip(ut_named.bonds(), ut_anon.bonds()):
        assert a == b


# ---- Tensor.Fromfile reads raw binary ----

def test_tensor_fromfile_reads_raw_binary(tmp_path):
    """Tensor.Fromfile must interpret the file as raw binary of the given dtype, not as a
    serialized .cytnx Tensor (the old binding mistakenly dispatched to Tensor::Load)."""
    fpath = str(tmp_path / "raw_doubles.bin")
    expected = np.array([1.0, 2.5, -3.25, 4.75], dtype=np.float64)
    with open(fpath, "wb") as f:
        f.write(expected.tobytes())

    t = cytnx.Tensor.Fromfile(fpath, cytnx.Type.Double, len(expected))
    assert t.shape() == [len(expected)]
    np.testing.assert_allclose(t.numpy().reshape(-1), expected)
