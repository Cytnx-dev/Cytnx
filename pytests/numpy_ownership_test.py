"""Regression tests for #941 ruling 4: zero-copy numpy views with capsule
ownership.

The old Storage.numpy()/Tensor.numpy() path called Storage.release() and
passed the raw pointer to py::array without a base object -- which BOTH
leaked the buffer (release() detached it from the destructor) AND still
copied (numpy cannot take ownership of a bare pointer). See the #941
discussion thread for the RSS-growth repro.

Now the bindings hand numpy a py::capsule owning a copy of the underlying
intrusive_ptr<Storage_base>: the array is a genuine zero-copy view and the
storage stays alive exactly as long as the array does. This is safe because
capacity_ was removed (ruling 2): every size-changing Storage operation
replaces the impl with a fresh buffer instead of mutating the allocation in
place, so a live numpy view is never resized out from under it.
"""

import gc

import numpy as np

import cytnx


def test_storage_numpy_zero_copy_sets_base():
    s = cytnx.Storage(5, cytnx.Type.Double)
    for i in range(5):
        s[i] = float(i)
    arr = s.numpy()
    assert arr.base is not None, "numpy() must set a base object (capsule keepalive)"
    assert not arr.flags.owndata, "numpy() must not copy"
    np.testing.assert_allclose(arr, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_storage_numpy_view_survives_storage_deletion():
    s = cytnx.Storage(5, cytnx.Type.Double)
    for i in range(5):
        s[i] = float(i)
    arr = s.numpy()
    del s
    gc.collect()
    # The capsule keeps the underlying StorageImplementation alive.
    np.testing.assert_allclose(arr, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_storage_numpy_shares_buffer():
    s = cytnx.Storage(3, cytnx.Type.Double)
    s.set_zeros()
    arr = s.numpy()
    arr[1] = 42.0
    assert s[1] == 42.0, "numpy view must alias the Storage buffer (zero-copy)"


def test_storage_numpy_roundtrip_dtypes():
    pairs = [
        (cytnx.Type.Double, np.float64),
        (cytnx.Type.Float, np.float32),
        (cytnx.Type.Int64, np.int64),
        (cytnx.Type.Uint64, np.uint64),
        (cytnx.Type.Int32, np.int32),
        (cytnx.Type.Uint32, np.uint32),
        (cytnx.Type.Bool, np.bool_),
        (cytnx.Type.ComplexDouble, np.complex128),
        (cytnx.Type.ComplexFloat, np.complex64),
    ]
    for cy_dtype, np_dtype in pairs:
        s = cytnx.Storage(3, cy_dtype)
        arr = s.numpy()
        assert arr.dtype == np_dtype, cytnx.Type.getname(cy_dtype)
        assert arr.shape == (3,)


def test_storage_numpy_repeated_alloc_no_corruption():
    for i in range(5):
        s = cytnx.Storage(1000, cytnx.Type.Double)
        arr = s.numpy()
        arr[:] = float(i)
        del s
        gc.collect()
        assert (arr == float(i)).all()


def test_tensor_numpy_default_is_independent_copy_without_leak():
    # share_mem=False (default): numpy gets a view of a fresh clone -- writing
    # to the array must NOT modify the original tensor (preserves the old
    # observable copy semantics, without the old leak).
    t = cytnx.zeros([2, 2])
    arr = t.numpy()
    assert arr.base is not None
    arr[0, 0] = 99.0
    assert t[0, 0].item() == 0.0


def test_tensor_numpy_share_mem_true_really_shares():
    # share_mem=True: previously this silently copied anyway (py::array with
    # no base copies); now it genuinely shares the buffer.
    t = cytnx.zeros([2, 2])
    arr = t.numpy(share_mem=True)
    arr[0, 0] = 7.5
    assert t[0, 0].item() == 7.5


def test_tensor_numpy_values_roundtrip():
    t = cytnx.arange(6).reshape(2, 3)
    arr = t.numpy()
    np.testing.assert_allclose(arr, [[0, 1, 2], [3, 4, 5]])
