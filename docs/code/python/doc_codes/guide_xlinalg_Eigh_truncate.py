# A Hermitian matrix with Z2 symmetry: one real symmetric block per charge sector,
# with eigenvalues of both signs.
bond = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0) >> 3, cytnx.Qs(1) >> 2], [cytnx.Symmetry.Zn(2)])
M = cytnx.UniTensor([bond, bond.redirect()], labels=["r", "c"], rowrank=1)
M.put_block_(cytnx.from_numpy(np.array([[5., 0., 0.], [0., -3., 0.], [0., 0., 0.5]])), 0)
M.put_block_(cytnx.from_numpy(np.array([[4., 0.], [0., 1.]])), 1)
# The full spectrum, block by block (ascending in each block)
D, V = cytnx.linalg.Eigh(M)
print([blk.numpy() for blk in D.get_blocks()])
# Keep the 3 eigenpairs of largest |eigenvalue| over both sectors: 5, 4, -3
D3, V3, err = cytnx.linalg.Eigh_truncate(M, keepdim=3, return_err=1)
print([blk.numpy() for blk in D3.get_blocks()])
print(D3.bonds()[0].getDegeneracies())  # kept dimension per sector
print(err.item())  # the largest discarded |eigenvalue|: 1
