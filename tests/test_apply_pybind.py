"""
Python-side smoke and correctness tests for UniTensor.apply() and UniTensor.apply_()
pybind bindings.

C++ reference: tests/BlockFermionicUniTensor_test.cpp (LinAlgElementwise test)
"""
import pytest
import cytnx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _are_nearly_eq(a, b, tol=1e-12):
    """Return True if two UniTensors have the same labels and nearly equal values."""
    a = a.permute(b.labels())
    blocks_a = a.get_blocks_()
    blocks_b = b.get_blocks_()
    if len(blocks_a) != len(blocks_b):
        return False
    for ba, bb in zip(blocks_a, blocks_b):
        if (ba - bb).Norm().item() > tol:
            return False
    return True


def _make_bfut3():
    """Construct the BFUT3 / BFUT3PERM fixture mirroring BlockFermionicUniTensorTest."""
    fp = cytnx.Symmetry.FermionParity()
    B1 = cytnx.Bond(cytnx.BD_IN,  [cytnx.Qs(0) >> 1, cytnx.Qs(1) >> 1], [fp])
    B2 = cytnx.Bond(cytnx.BD_IN,  [cytnx.Qs(0) >> 1, cytnx.Qs(1) >> 1], [fp])
    B12 = B1.combineBond(B2).c_redirect_()
    B3 = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(0) >> 1, cytnx.Qs(1) >> 1], [fp])
    B4 = cytnx.Bond(cytnx.BD_IN,  [cytnx.Qs(0) >> 1, cytnx.Qs(1) >> 1], [fp])

    BFUT3 = cytnx.UniTensor([B1, B2, B12, B3, B4], labels=["a", "b", "c", "d", "e"])
    BFUT3.set_elem([0, 0, 0, 0, 0], 1.0)
    BFUT3.set_elem([0, 0, 1, 0, 0], 2.0)
    BFUT3.set_elem([0, 1, 2, 0, 0], 3.0)
    BFUT3.set_elem([0, 1, 3, 0, 0], 4.0)
    BFUT3.set_elem([1, 0, 2, 0, 0], 5.0)
    BFUT3.set_elem([1, 0, 3, 0, 0], 6.0)
    BFUT3.set_elem([1, 1, 0, 0, 0], 7.0)
    BFUT3.set_elem([1, 1, 1, 0, 0], 8.0)

    BFUT3PERM = cytnx.UniTensor([B3, B2, B4, B12, B1], labels=["d", "b", "e", "c", "a"])
    BFUT3PERM.set_elem([0, 0, 0, 0, 0],  1.0)
    BFUT3PERM.set_elem([0, 0, 0, 1, 0],  2.0)
    BFUT3PERM.set_elem([0, 1, 0, 2, 0],  3.0)
    BFUT3PERM.set_elem([0, 1, 0, 3, 0],  4.0)
    BFUT3PERM.set_elem([0, 0, 0, 2, 1], -5.0)
    BFUT3PERM.set_elem([0, 0, 0, 3, 1], -6.0)
    BFUT3PERM.set_elem([0, 1, 0, 0, 1], -7.0)
    BFUT3PERM.set_elem([0, 1, 0, 1, 1], -8.0)

    T = BFUT3.permute([3, 1, 4, 2, 0]).contiguous()
    return T, BFUT3PERM


# ---------------------------------------------------------------------------
# DenseUniTensor: apply() and apply_() are no-ops
# ---------------------------------------------------------------------------

class TestApplyDenseUniTensor:
    def setup_method(self):
        self.ut = cytnx.UniTensor(cytnx.zeros([2, 3]))
        self.ut.set_elem([0, 1], 1.0)
        self.ut.set_elem([1, 2], 2.0)

    def test_apply_returns_unitensor(self):
        assert isinstance(self.ut.apply(), cytnx.UniTensor)

    def test_apply_is_noop(self):
        result = self.ut.apply()
        diff = result.get_block() - self.ut.get_block()
        assert diff.Norm().item() < 1e-14

    def test_apply_inplace_returns_unitensor(self):
        assert isinstance(self.ut.apply_(), cytnx.UniTensor)

    def test_apply_inplace_is_noop(self):
        original = self.ut.get_block().clone()
        self.ut.apply_()
        assert (self.ut.get_block() - original).Norm().item() < 1e-14


# ---------------------------------------------------------------------------
# BlockFermionicUniTensor: apply() applies pending signflips
# ---------------------------------------------------------------------------

class TestApplyBlockFermionicUniTensor:
    def setup_method(self):
        self.T, self.BFUT3PERM = _make_bfut3()

    def test_apply_returns_unitensor(self):
        assert isinstance(self.T.apply(), cytnx.UniTensor)

    def test_apply_applies_signflips(self):
        result = self.T.apply()
        assert _are_nearly_eq(result, self.BFUT3PERM)

    def test_apply_does_not_modify_original(self):
        clone = self.T.clone()
        self.T.apply()
        assert _are_nearly_eq(self.T.apply(), clone.apply())

    def test_apply_inplace_applies_signflips(self):
        self.T.apply_()
        assert _are_nearly_eq(self.T, self.BFUT3PERM)

    def test_apply_inplace_returns_unitensor(self):
        result = self.T.apply_()
        assert isinstance(result, cytnx.UniTensor)
