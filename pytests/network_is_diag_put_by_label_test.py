"""Regression tests for Cytnx-dev/Cytnx#1207.

A Z_N-symmetric ``is_diag=True`` UniTensor put into a ``Network`` with the
by-label form ``PutUniTensor(name, T, labels)`` used to crash ``Launch()`` with
SIGSEGV: the label permutation produced a diagonal BlockUniTensor without its
blocks. The expected values are the products of the diagonal entries.
"""

import pytest

import cytnx


def _diag_and_full(n=3):
    sym = [cytnx.Symmetry.Zn(n)]
    qs = [cytnx.Qs(q) >> 1 for q in range(n)]
    bond_out = cytnx.Bond(cytnx.BD_OUT, qs, sym)
    bond_in = bond_out.redirect()
    W = cytnx.UniTensor([bond_in, bond_out], labels=["i", "x"], rowrank=1, is_diag=True)
    A = cytnx.UniTensor([bond_in, bond_out], labels=["x", "o"], rowrank=1)
    for q in range(n):
        W.at(["i", "x"], [q, q]).value = float(q + 1)
        A.at(["x", "o"], [q, q]).value = float(2 * q + 1)
    return W, A


def _launch(put_labels):
    W, A = _diag_and_full()
    net = cytnx.Network()
    net.FromString(["W: i, x", "A: x, o", "TOUT: i; o"])
    net.PutUniTensor("A", A)
    if put_labels is None:
        net.PutUniTensor("W", W)
    else:
        net.PutUniTensor("W", W, put_labels)
    out = net.Launch()
    return [blk.numpy().ravel().tolist() for blk in out.get_blocks()]


EXPECTED = [[1.0 * 1.0], [2.0 * 3.0], [3.0 * 5.0]]


def test_symmetric_is_diag_put_by_label_launches():
    assert _launch(["i", "x"]) == EXPECTED


def test_symmetric_is_diag_put_by_label_in_permuted_order_launches():
    W, A = _diag_and_full()
    net = cytnx.Network()
    net.FromString(["W: x, i", "A: x, o", "TOUT: i; o"])
    net.PutUniTensor("A", A)
    net.PutUniTensor("W", W, ["x", "i"])
    out = net.Launch()
    assert [blk.numpy().ravel().tolist() for blk in out.get_blocks()] == EXPECTED


def test_symmetric_is_diag_two_argument_put_agrees():
    assert _launch(None) == EXPECTED


def test_permute_of_symmetric_is_diag_keeps_its_blocks():
    W, _ = _diag_and_full()
    P = W.permute(["x", "i"])
    assert P.is_diag()
    assert P.Nblocks() == W.Nblocks()
    assert [blk.numpy().tolist() for blk in P.get_blocks()] == \
        [blk.numpy().tolist() for blk in W.get_blocks()]
