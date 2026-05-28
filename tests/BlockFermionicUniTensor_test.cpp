#include "BlockFermionicUniTensor_test.h"

#include <algorithm>

// Helper: verify that X is an isometry / unitary over its auxiliary index `aux`: contracting the
// physical legs against X^dagger yields the identity on the aux index. Handles both conventions:
//   - aux is the FIRST leg (e.g. Svd vT):        checks X X^dagger = I
//   - aux is the LAST  leg (e.g. Svd/Eig U, V):  checks X^dagger X = I
// Per the fermionic contraction rule, the dagger operand is followed by fermion_twists() and placed
// on the left of the contraction.
static void expect_unitary(const UniTensor &X, const std::string &aux, double tol) {
  UniTensor Xd = X.Dagger();
  Xd.relabel_(aux, aux + "_dag");
  UniTensor G = (X.labels().front() == aux) ? Contract(X.fermion_twists(), Xd)
                                            : Contract(Xd.fermion_twists(), X);
  UniTensor Id = 0. * G.clone();
  for (cytnx_uint64 k = 0; k < Id.shape()[0]; k++) {
    auto p = Id.at({k, k});
    p = 1.0;
  }
  EXPECT_TRUE((G.apply() - Id.apply()).Norm().item() < tol);
}

// Helper: verify an SVD decomposition usv = [S, U, vT]: U and vT are isometries (U^dagger U = I,
// vT vT^dagger = I) and the reconstruction A = U S vT holds. vT is the genuine
// right-singular-vector tensor (already living on the column space), so the reconstruction is the
// direct product -- no Dagger construction and no fermion_twists are needed (this is what
// distinguishes Svd/Gesvd from the eigen-decompositions, where only V is returned).
static void expect_svd_reconstructs(const std::vector<UniTensor> &usv, const UniTensor &A,
                                    double tol) {
  ASSERT_EQ(usv.size(), 3);
  UniTensor S = usv[0], U = usv[1], vT = usv[2];
  expect_unitary(U, "_aux_L", tol);
  expect_unitary(vT, "_aux_R", tol);
  UniTensor recon = Contract(Contract(U, S), vT);
  recon.permute_(A.labels());
  EXPECT_TRUE((recon.apply() - A.apply()).Norm().item() < tol);
}

/*=====test info=====
describe:scalar product between two vectors
====================*/
TEST_F(BlockFermionicUniTensorTest, VectorContract) {
  // 1+2*2-3*3-4*4-5*5-6*6+7*7+8*8 = 32
  EXPECT_TRUE(abs(BFUT1.contract(BFUT2).item() - 32) < 1e-12);
}

/*=====test info=====
describe:contraction
====================*/
TEST_F(BlockFermionicUniTensorTest, SimpleTensorContract) {
  // 1+2*2-3*3-4*4-5*5-6*6+7*7+8*8 = 32
  EXPECT_TRUE(abs(BFUT3.contract(BFUT2).at({0, 0}) - 32) < 1e-13);
}

/*=====test info=====
describe:some elementwise linear algebra functions
====================*/
TEST_F(BlockFermionicUniTensorTest, LinAlgElementwise) {
  const double tol = 1e-14;
  UniTensor T = BFUT3.permute({3, 1, 4, 2, 0}).contiguous();
  EXPECT_TRUE(AreEqUniTensor(BFUT3PERM, T.apply()));
  UniTensor res = T + T;
  EXPECT_TRUE(AreNearlyEqUniTensor(2. * BFUT3PERM, res.apply_(), tol));
  res = T + BFUT3PERM;
  EXPECT_TRUE(AreNearlyEqUniTensor(2. * BFUT3PERM, res.apply_(), tol));
  res = (T + T + T + T) / 4.;
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), BFUT3PERM, tol));
  res = (T + T + BFUT3PERM + T) / 4.;
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), BFUT3PERM, tol));
  res = (2 * T) - T;
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), BFUT3PERM, tol));
  res = (2 * T) - BFUT3PERM;
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), BFUT3PERM, tol));
  res = BFUT3PERM * BFUT3PERM;
  UniTensor ref = T * T;
  res.permute_(ref.labels());
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), ref.apply_(), tol));
  res = (T * T * T) / T;
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), BFUT3PERM * BFUT3PERM, tol));
  // negation
  res = BFUT3PERM * (-1. * BFUT3PERM);
  ref = -1 * ref;
  ref.permute_(res.labels());
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), ref.apply_(), tol));
  res = (-1. * BFUT3PERM) * BFUT3PERM;
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), ref, tol));
  // commutative property
  res = BFUT3PERM.permute_nosignflip(T.labels()) * T;
  ref = T * BFUT3PERM.permute_nosignflip(T.labels());
  ref.permute_(res.labels());
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), ref.apply_(), tol));
  res = BFUT3PERM.permute_nosignflip(T.labels()).permute_(BFUT3PERM.labels()) * BFUT3PERM;
  ref = BFUT3PERM * BFUT3PERM.permute_nosignflip(T.labels()).permute_(BFUT3PERM.labels());
  ref.permute_(res.labels());
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), ref.apply_(), tol));
  // test inline
  res = T;
  res += T;
  res /= 2.;
  EXPECT_EQ(AreNearlyEqUniTensor(res.apply_(), T.apply(), tol), true);
  // test Mul and Div for tensors
  res = T.clone();
  res *= T;
  ref = T * T;
  EXPECT_EQ(AreNearlyEqUniTensor(res.apply_(), ref.apply_(), tol), true);
  res /= T;
  EXPECT_EQ(AreNearlyEqUniTensor(res.apply_(), T.apply(), tol), true);
  // check power, only works for unpermuted tensor (otherwise, sign structure of T*T and T.Pow(2.)
  // differs)
  res = BFUT3.clone();
  res *= BFUT3;
  EXPECT_EQ(AreNearlyEqUniTensor(res.apply_(), BFUT3.Pow(2.).apply(), tol), true);
}

/*=====test info=====
describe:test fermion_twists behavior on tagged fermionic tensors
====================*/
TEST_F(BlockFermionicUniTensorTest, FermionTwists) {
  UniTensor twisted = BFUT5.fermion_twists();
  UniTensor manual = BFUT5.clone();
  for (cytnx_int64 idx = manual.rowrank(); idx < manual.rank(); idx++) {
    if (manual.bonds()[idx].type() != BD_BRA) manual.twist_(idx);
  }
  EXPECT_EQ(twisted.signflip(), manual.signflip());
  EXPECT_TRUE(AreEqUniTensor(twisted.apply_(), manual.apply_()));

  // applying fermion_twists_ twice toggles the same set of signs twice
  UniTensor twice = BFUT5.clone();
  twice.fermion_twists_().fermion_twists_();
  EXPECT_EQ(twice.signflip(), BFUT5.signflip());
  EXPECT_TRUE(AreEqUniTensor(twice.apply_(), BFUT5.apply()));
}

/*=====test info=====
describe:test SVD unitarity and reconstruction for fermionic tensors with mixed in/out legs
====================*/
TEST_F(BlockFermionicUniTensorTest, SvdUnitaryAndReconstruction) {
  const double tol = 1e-10;
  // U, vT are isometries and BFUT5 = U S vT (direct SVD reconstruction).
  expect_svd_reconstructs(linalg::Svd_truncate(BFUT5, 1000, 0., true, 0), BFUT5, tol);
}

TEST_F(BlockFermionicUniTensorTest, group_basis) {
  auto out = BFUT4.group_basis();

  // careful: the quantum numbers are sorted; the last index such changes from (0,1,2) to (1,2,0)
  EXPECT_DOUBLE_EQ(double(out.at({0, 0, 0}).real()), double(1));
  EXPECT_DOUBLE_EQ(double(out.at({0, 0, 1}).real()), double(2));
  EXPECT_DOUBLE_EQ(double(out.at({0, 1, 2}).real()), double(3));
  EXPECT_DOUBLE_EQ(double(out.at({1, 0, 2}).real()), double(4));
  EXPECT_DOUBLE_EQ(double(out.at({2, 0, 2}).real()), double(5));
  EXPECT_DOUBLE_EQ(double(out.at({1, 1, 0}).real()), double(6));
  EXPECT_DOUBLE_EQ(double(out.at({2, 1, 0}).real()), double(7));
  EXPECT_DOUBLE_EQ(double(out.at({1, 1, 1}).real()), double(8));
  EXPECT_DOUBLE_EQ(double(out.at({2, 1, 1}).real()), double(9));

  out = BFUT4.permute({1, 0, 2});
  out = out.group_basis();
  out.permute_(BFUT4.labels());
  out.apply_();
  // the block indices can differ, therefore I compare the elements
  EXPECT_DOUBLE_EQ(double(out.at({0, 0, 0}).real()), double(1));
  EXPECT_DOUBLE_EQ(double(out.at({0, 0, 1}).real()), double(2));
  EXPECT_DOUBLE_EQ(double(out.at({0, 1, 2}).real()), double(3));
  EXPECT_DOUBLE_EQ(double(out.at({1, 0, 2}).real()), double(4));
  EXPECT_DOUBLE_EQ(double(out.at({2, 0, 2}).real()), double(5));
  EXPECT_DOUBLE_EQ(double(out.at({1, 1, 0}).real()), double(6));
  EXPECT_DOUBLE_EQ(double(out.at({2, 1, 0}).real()), double(7));
  EXPECT_DOUBLE_EQ(double(out.at({1, 1, 1}).real()), double(8));
  EXPECT_DOUBLE_EQ(double(out.at({2, 1, 1}).real()), double(9));

  out = BFUT4.permute({1, 2, 0});
  out = out.group_basis();
  out.permute_(BFUT4.labels());
  out.apply_();
  // the block indices can differ, therefore I compare the elements
  EXPECT_DOUBLE_EQ(double(out.at({0, 0, 0}).real()), double(1));
  EXPECT_DOUBLE_EQ(double(out.at({0, 0, 1}).real()), double(2));
  EXPECT_DOUBLE_EQ(double(out.at({0, 1, 2}).real()), double(3));
  EXPECT_DOUBLE_EQ(double(out.at({1, 0, 2}).real()), double(4));
  EXPECT_DOUBLE_EQ(double(out.at({2, 0, 2}).real()), double(5));
  EXPECT_DOUBLE_EQ(double(out.at({1, 1, 0}).real()), double(6));
  EXPECT_DOUBLE_EQ(double(out.at({2, 1, 0}).real()), double(7));
  EXPECT_DOUBLE_EQ(double(out.at({1, 1, 1}).real()), double(8));
  EXPECT_DOUBLE_EQ(double(out.at({2, 1, 1}).real()), double(9));
}

/*=====test info=====
describe:test pseudo-inverse
====================*/
TEST_F(BlockFermionicUniTensorTest, Inv) {
  const double tol = 1e-14;
  double clip = 1e-15;
  EXPECT_TRUE(AreNearlyEqUniTensor(BFUT3.Inv(clip), BFUT3INV, tol));
  UniTensor T = BFUT3.permute({3, 1, 4, 2, 0}).contiguous();
  EXPECT_TRUE(AreNearlyEqUniTensor(T.Inv(clip).Inv_(clip), T, tol));
  EXPECT_FALSE(AreNearlyEqUniTensor(T.Inv(clip), T, tol));
  clip = 0.1;  // test actual clipping as well
  auto tmp = T.clone();
  tmp.Inv_(clip);  // test inline version
  EXPECT_TRUE(AreEqUniTensor(T.Inv(clip), tmp));
  tmp = T.clone();
  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 4; k++)
        for (size_t l = 0; l < 2; l++)
          for (size_t m = 0; m < 2; m++) {
            // elements are permuted!
            auto proxy = tmp.at({l, j, m, k, i});
            if (proxy.exists()) {
              Scalar val = proxy;
              if (val.abs() <= clip)
                proxy = 0.;
              else
                proxy = 1. / proxy;
            }
          }
  EXPECT_TRUE(AreNearlyEqUniTensor(T.Inv(clip), tmp, tol));
}

/*=====test info=====
describe:test power
====================*/
TEST_F(BlockFermionicUniTensorTest, Pow) {
  const double tol = 1e-14;
  UniTensor T = BFUT3.permute({3, 1, 4, 2, 0}).contiguous();
  // only even powers are products, odd powers differ by signflips
  EXPECT_TRUE(AreNearlyEqUniTensor(T.Pow(3.), T * T * T, tol));
  auto tmp = T.clone();
  tmp.Pow_(2.3);  // test inline version
  EXPECT_TRUE(AreEqUniTensor(T.Pow(2.3), tmp));
  for (double p = 0.; p < 1.6; p += 0.5) {
    tmp = T.clone();
    for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 2; j++)
        for (size_t k = 0; k < 4; k++)
          for (size_t l = 0; l < 2; l++)
            for (size_t m = 0; m < 2; m++) {
              // elements are permuted!
              auto proxy = tmp.at({l, j, m, k, i});
              if (proxy.exists()) {
                Scalar val = proxy;
                proxy = std::pow((cytnx_double)val, p);
              }
            }
    EXPECT_TRUE(AreNearlyEqUniTensor(T.Pow(p), tmp, tol));
  }
}

/*=====test info=====
describe:write to disc
====================*/
TEST_F(BlockFermionicUniTensorTest, SaveLoad) {
  BFUT1.Save(temp_file_path);
  UniTensor BFUT1_loaded = BFUT1_loaded.Load(temp_file_path);
  EXPECT_TRUE(AreEqUniTensor(BFUT1, BFUT1_loaded));
  // for char*
  const char *fname = temp_file_path.c_str();
  BFUT1.Save(fname);
  UniTensor BFUT1_loaded_char_save = BFUT1_loaded_char_save.Load(temp_file_path);
  EXPECT_TRUE(AreEqUniTensor(BFUT1, BFUT1_loaded_char_save));
  UniTensor BFUT1_loaded_char_load = BFUT1_loaded_char_load.Load(fname);
  EXPECT_TRUE(AreEqUniTensor(BFUT1, BFUT1_loaded_char_load));
}

/*=====test info=====
describe:test Transpose and Transpose_ for BlockFermionicUniTensor:
  rowrank is updated, index order is reversed, bonds are redirected,
  and element values are preserved without sign flips.
====================*/
TEST_F(BlockFermionicUniTensorTest, Transpose) {
  // BFUT1: rank=3, rowrank=2, bonds=[BD_IN(a), BD_IN(b), BD_OUT(c)], shape=(2,2,4)
  EXPECT_EQ(BFUT1.rowrank(), 2);

  auto tmp = BFUT1.Transpose();

  // rowrank must be rank - old_rowrank = 3 - 2 = 1
  EXPECT_EQ(tmp.rowrank(), 1);
  EXPECT_EQ(tmp.rank(), 3);

  // index order is reversed: new [0,1,2] = old [c,b,a]
  EXPECT_EQ(tmp.labels()[0], "c");
  EXPECT_EQ(tmp.labels()[1], "b");
  EXPECT_EQ(tmp.labels()[2], "a");

  // bonds are redirected: old BD_OUT(c)->BD_IN, old BD_IN(b)->BD_OUT, old BD_IN(a)->BD_OUT
  EXPECT_EQ(tmp.bonds()[0].type(), BD_IN);
  EXPECT_EQ(tmp.bonds()[1].type(), BD_OUT);
  EXPECT_EQ(tmp.bonds()[2].type(), BD_OUT);

  // element at old {a,b,c} appears at new {c,b,a}; no sign flips
  EXPECT_DOUBLE_EQ(double(tmp.at({0, 0, 0}).real()), 1.);
  EXPECT_DOUBLE_EQ(double(tmp.at({1, 0, 0}).real()), 2.);
  EXPECT_DOUBLE_EQ(double(tmp.at({2, 1, 0}).real()), 3.);
  EXPECT_DOUBLE_EQ(double(tmp.at({3, 1, 0}).real()), 4.);
  EXPECT_DOUBLE_EQ(double(tmp.at({2, 0, 1}).real()), 5.);
  EXPECT_DOUBLE_EQ(double(tmp.at({3, 0, 1}).real()), 6.);
  EXPECT_DOUBLE_EQ(double(tmp.at({0, 1, 1}).real()), 7.);
  EXPECT_DOUBLE_EQ(double(tmp.at({1, 1, 1}).real()), 8.);

  // Transpose is an involution: T.Transpose().Transpose() == T
  EXPECT_TRUE(AreEqUniTensor(tmp.Transpose(), BFUT1));

  // in-place version must match
  auto tmp2 = BFUT1.clone();
  tmp2.Transpose_();
  EXPECT_EQ(tmp2.rowrank(), 1);
  EXPECT_EQ(tmp2.bonds()[0].type(), BD_IN);
  EXPECT_EQ(tmp2.bonds()[1].type(), BD_OUT);
  EXPECT_EQ(tmp2.bonds()[2].type(), BD_OUT);
  EXPECT_TRUE(AreEqUniTensor(tmp2, tmp));
}

// Helper: square fermionic UniTensor (combined row space == combined column space)
// with one even (Qs 0) and one odd (Qs 1) sector, each of degeneracy 2.
static UniTensor make_square_fermionic(const std::vector<std::string> &labels) {
  Bond Bi = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 2}, {Symmetry::FermionParity()});
  Bond Bo = Bi.redirect();
  return UniTensor({Bi, Bo}, labels);
}

// Helper: rank-4 square fermionic UniTensor on row space (a,b), canonical leg order [a, b, a*, b*]
// (rowrank 2). Real-symmetric values -> Hermitian. A *consistent* permute (same permutation on
// row and column legs, e.g. {1,0,3,2}) then makes the pending signflip non-trivial while leaving
// the operator (hence its spectrum / singular values) unchanged.
static UniTensor make_rank4_hermitian() {
  Bond a = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
  Bond b = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
  UniTensor M = UniTensor({a, b, a.redirect(), b.redirect()}, {"a", "b", "c", "d"});
  M.set_rowrank_(2);
  auto sh = M.shape();
  for (cytnx_uint64 i = 0; i < sh[0]; i++)
    for (cytnx_uint64 j = 0; j < sh[1]; j++)
      for (cytnx_uint64 k = 0; k < sh[2]; k++)
        for (cytnx_uint64 l = 0; l < sh[3]; l++) {
          auto p = M.at({i, j, k, l});
          if (!p.exists()) continue;
          cytnx_uint64 r = 2 * i + j, c = 2 * k + l;
          cytnx_uint64 lo = std::min(r, c), hi = std::max(r, c);
          p = 0.1 * double(1 + lo * 4 + hi) +
              0.2;  // depends only on the unordered {r,c} -> Hermitian
        }
  return M;
}

// Helper: collect the diagonal entries (real parts) of a (block-)diagonal UniTensor, sorted
// ascending. Eigenvalues / singular values are invariant under a consistent permute, so comparing
// these between a tensor and its permuted (sign-flip-active) form verifies correct sign handling.
static std::vector<double> sorted_diagonal(UniTensor d) {
  std::vector<double> vals;
  auto sh = d.shape();
  for (cytnx_uint64 k = 0; k < sh[0]; k++) {
    auto p = d.at({k, k});
    if (p.exists()) vals.push_back(double(p.real()));
  }
  std::sort(vals.begin(), vals.end());
  return vals;
}

// Helper: assert two (block-)diagonal tensors carry the same (real) diagonal multiset. Used to
// cross-check spectra/singular values across solvers (Eigh vs Eig, Gesvd vs Svd) and across a
// consistent sign-flip permute.
static void expect_same_diagonal(const UniTensor &a, const UniTensor &b, double tol) {
  auto va = sorted_diagonal(a), vb = sorted_diagonal(b);
  ASSERT_EQ(va.size(), vb.size());
  for (size_t i = 0; i < va.size(); i++) EXPECT_NEAR(va[i], vb[i], tol);
}

// Helper: build a permuted (consistent {1,0,3,2}) copy and assert it carries non-trivial sign
// flips.
static UniTensor permute_with_signflips(const UniTensor &M) {
  UniTensor Mp = M.permute({1, 0, 3, 2}).contiguous();
  bool anyflip = false;
  for (auto f : Mp.signflip()) anyflip = anyflip || f;
  EXPECT_TRUE(anyflip);  // ensure the signflip negation path is actually exercised
  return Mp;
}

/*=====test info=====
describe:Eigh for fermionic tensors: real spectrum, orthonormal eigenvectors
  (V^dagger V = I), and spectrum invariance under a consistent sign-flip permute.
====================*/
TEST_F(BlockFermionicUniTensorTest, Eigh) {
  const double tol = 1e-10;
  UniTensor H = make_square_fermionic({"i", "o"});
  // real symmetric per sector -> Hermitian fermionic UniTensor
  H.at({0, 0}) = 2.0;
  H.at({0, 1}) = 0.7;
  H.at({1, 0}) = 0.7;
  H.at({1, 1}) = 3.0;
  H.at({2, 2}) = 1.5;
  H.at({2, 3}) = -0.4;
  H.at({3, 2}) = -0.4;
  H.at({3, 3}) = 2.5;

  auto out = linalg::Eigh(H);
  ASSERT_EQ(out.size(), 2);
  UniTensor e = out[0], V = out[1];
  EXPECT_TRUE(e.is_diag());
  EXPECT_EQ(e.dtype(), Type.Double);  // Eigh eigenvalues are real

  // eigenvectors are orthonormal: V^dagger V = I
  expect_unitary(V, "_aux_L", tol);

  // cross-check against the general solver: for a Hermitian operator, Eig must return the same
  // (real) spectrum as Eigh (Eig eigenvalues are complex; sorted_diagonal compares real parts).
  expect_same_diagonal(e, linalg::Eig(H)[0], tol);

  // sign-flip coverage: a consistent permute makes the pending signflip non-trivial (exercising
  // the negation path in Eigh_BlockFermionic_UT_internal). Verify the eigenvectors are still
  // orthonormal (V^dagger V = I) and the spectrum is unchanged.
  UniTensor M = make_rank4_hermitian();
  UniTensor Mp = permute_with_signflips(M);
  auto outp = linalg::Eigh(Mp);
  UniTensor ep = outp[0], Vp = outp[1];
  expect_unitary(Vp, "_aux_L", tol);
  expect_same_diagonal(ep, linalg::Eigh(M)[0], tol);

  // TODO: issue #782: test reconstruction of M from U^-1 E U once Eig/Eigh return the inverse
  // eigenvector tensor U^-1 (which equals U^dagger only in the unitary/Hermitian case, and in
  // general carries bonds inherited from the column space of the input tensor).
}

/*=====test info=====
describe:Eig for fermionic tensors: complex spectrum and the eigen equation
  A V = V e (column-vector / row_v=false convention).
====================*/
TEST_F(BlockFermionicUniTensorTest, Eig) {
  const double tol = 1e-10;
  UniTensor A = make_square_fermionic({"i", "o"});
  A.at({0, 0}) = 1.0;
  A.at({0, 1}) = 2.0;
  A.at({1, 0}) = 0.5;
  A.at({1, 1}) = 1.3;
  A.at({2, 2}) = 2.0;
  A.at({2, 3}) = 0.4;
  A.at({3, 2}) = 0.9;
  A.at({3, 3}) = 3.0;

  auto out = linalg::Eig(A);
  ASSERT_EQ(out.size(), 2);
  UniTensor e = out[0], V = out[1];
  EXPECT_TRUE(e.is_diag());
  EXPECT_EQ(e.dtype(), Type.ComplexDouble);  // general (non-Hermitian) eigenvalues are complex

  // eigen equation: A V = V e. NOTE: this check relies on A having a SINGLE column leg ("o").
  // Relabeling V's row leg onto the column leg ("i" -> "o") and contracting then carries no
  // fermionic reorder sign. With two or more column legs (e.g. the rank-4 case below) the same
  // construction would have to swap an odd-parity column-leg pair, which picks up a sign that no
  // fermion_twists() placement cancels -- so the eigen equation is only verified here, and the
  // rank-4 sign-flip case is covered by spectrum invariance instead (full reconstruction: issue
  // #782).
  UniTensor Ac = A.astype(Type.ComplexDouble);
  UniTensor Vo = V.clone();
  Vo.relabel_("i", "o");
  UniTensor AV = Contract(Ac.fermion_twists(), Vo);
  UniTensor Ve = Contract(V, e);
  Ve.relabel_("_aux_R", "_aux_L");
  AV.permute_(std::vector<std::string>{"i", "_aux_L"});
  Ve.permute_(std::vector<std::string>{"i", "_aux_L"});
  EXPECT_TRUE((AV.apply() - Ve.apply()).Norm().item() < tol);

  // TODO: issue #782: test reconstruction of M from U^-1 E U once Eig/Eigh return the inverse
  // eigenvector tensor U^-1 (which equals U^dagger only in the unitary/Hermitian case, and in
  // general carries bonds inherited from the column space of the input tensor).

  // sign-flip coverage: a consistent permute of a (Hermitian, hence real-spectrum) operator
  // leaves the eigenvalues invariant but makes the pending signflip non-trivial, exercising the
  // negation path in Eig_BlockFermionic_UT_internal.
  UniTensor M = make_rank4_hermitian();
  UniTensor Mp = permute_with_signflips(M);
  expect_same_diagonal(linalg::Eig(M)[0], linalg::Eig(Mp)[0], tol);
}

/*=====test info=====
describe:Qr for fermionic tensors: reconstruction Q R = A, including a rank-4
  case with non-trivial fermionic sign flips.
====================*/
TEST_F(BlockFermionicUniTensorTest, Qr) {
  const double tol = 1e-10;
  UniTensor A = make_square_fermionic({"i", "o"});
  A.at({0, 0}) = 1.0;
  A.at({0, 1}) = 2.0;
  A.at({1, 0}) = 0.5;
  A.at({1, 1}) = 1.3;
  A.at({2, 2}) = 2.0;
  A.at({2, 3}) = 0.4;
  A.at({3, 2}) = 0.9;
  A.at({3, 3}) = 3.0;

  auto out = linalg::Qr(A);
  ASSERT_EQ(out.size(), 2);
  expect_unitary(out[0], "_aux_", tol);  // Q is an isometry: Q^dagger Q = I
  UniTensor recon = Contract(out[0], out[1]);
  recon.permute_(A.labels());
  EXPECT_TRUE((recon.apply() - A.apply()).Norm().item() < tol);

  // rank-4 case whose fermionic reordering produces non-trivial sign flips,
  // exercising the signflip negation path in Qr_BlockFermionic_UT_internal.
  Bond ba = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
  Bond bb = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
  UniTensor T = UniTensor({ba, bb, ba.redirect(), bb.redirect()}, {"a", "b", "c", "d"});
  T.set_rowrank_(2);
  cytnx_double val = 1.0;
  auto sh = T.shape();
  for (cytnx_uint64 i = 0; i < sh[0]; i++)
    for (cytnx_uint64 j = 0; j < sh[1]; j++)
      for (cytnx_uint64 k = 0; k < sh[2]; k++)
        for (cytnx_uint64 l = 0; l < sh[3]; l++) {
          auto proxy = T.at({i, j, k, l});
          if (proxy.exists()) {
            proxy = val + 0.3;
            val += 1.0;
          }
        }
  UniTensor Tp = T.permute({1, 0, 2, 3}).contiguous();
  bool anyflip = false;
  for (auto f : Tp.signflip()) anyflip = anyflip || f;
  EXPECT_TRUE(anyflip);  // guard: ensure the sign-flip code path is actually exercised

  auto out4 = linalg::Qr(Tp);
  ASSERT_EQ(out4.size(), 2);
  expect_unitary(out4[0], "_aux_", tol);  // Q is an isometry even with sign flips active
  UniTensor recon4 = Contract(out4[0], out4[1]);
  recon4.permute_(Tp.labels());
  EXPECT_TRUE((recon4.apply() - Tp.apply()).Norm().item() < tol);
}

/*=====test info=====
describe:Eig/Eigh argument guards for fermionic tensors: row_v=true is rejected
  only when eigenvectors are requested (is_V), and rowrank must be < rank.
====================*/
TEST_F(BlockFermionicUniTensorTest, EigEighRowVGuards) {
  UniTensor H = make_square_fermionic({"i", "o"});
  H.at({0, 0}) = 2.0;
  H.at({0, 1}) = 0.7;
  H.at({1, 0}) = 0.7;
  H.at({1, 1}) = 3.0;
  H.at({2, 2}) = 1.5;
  H.at({2, 3}) = -0.4;
  H.at({3, 2}) = -0.4;
  H.at({3, 3}) = 2.5;

  // row_v=true is unsupported, but only when eigenvectors are requested.
  EXPECT_THROW({ linalg::Eig(H, true, true); }, std::logic_error);
  EXPECT_THROW({ linalg::Eigh(H, true, true); }, std::logic_error);
  // eigenvalue-only calls must not be rejected by the row_v flag.
  EXPECT_NO_THROW({ linalg::Eig(H, false, true); });
  EXPECT_NO_THROW({ linalg::Eigh(H, false, true); });

  // rowrank must be strictly less than rank.
  UniTensor sq = make_square_fermionic({"i", "o"});
  sq.set_rowrank_(2);
  EXPECT_THROW({ linalg::Eig(sq); }, std::logic_error);
  EXPECT_THROW({ linalg::Eigh(sq); }, std::logic_error);
}

/*=====test info=====
describe:test Gesvd_truncate (?gesvd-based SVD) unitarity and reconstruction for
  fermionic tensors with mixed in/out legs. Mirrors SvdUnitaryAndReconstruction.
====================*/
TEST_F(BlockFermionicUniTensorTest, GesvdTruncateUnitaryAndReconstruction) {
  const double tol = 1e-10;
  // mixed in/out legs (no pending signflip)
  auto gesvd = linalg::Gesvd_truncate(BFUT5, 1000, 0., true, true, 0);
  expect_svd_reconstructs(gesvd, BFUT5, tol);
  // cross-check against Svd: ?gesvd and ?gesdd must yield the same singular values.
  expect_same_diagonal(gesvd[0], linalg::Svd_truncate(BFUT5, 1000, 0., true, 0)[0], tol);

  // sign-flip-active: a consistent permute makes the pending signflip non-trivial, exercising the
  // negation path in the BlockFermionic truncation while leaving U, S, vT correct.
  UniTensor Msf = permute_with_signflips(make_rank4_hermitian());
  auto gesvd_sf = linalg::Gesvd_truncate(Msf, 1000, 0., true, true, 0);
  expect_svd_reconstructs(gesvd_sf, Msf, tol);
  expect_same_diagonal(gesvd_sf[0], linalg::Svd_truncate(Msf, 1000, 0., true, 0)[0], tol);
}
