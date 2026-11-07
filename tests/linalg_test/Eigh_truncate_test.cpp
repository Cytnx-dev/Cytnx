#include <gtest/gtest.h>

#include "../test_tools.h"
#include "cytnx.hpp"
#include "linalg_test.h"

namespace cytnx {
  namespace test {
    namespace {

      // A rank-2 Hermitian BlockUniTensor whose block in each Z_N sector is the given real
      // symmetric matrix, with labels ("r", "c") and bond directions (IN, OUT).
      UniTensor hermitian_blocks(unsigned int n_sectors,
                                 const std::vector<std::vector<std::vector<double>>> &blocks) {
        std::vector<Qs> qs;
        std::vector<cytnx_uint64> degs;
        for (unsigned int q = 0; q < n_sectors; q++) {
          qs.push_back(Qs(static_cast<cytnx_int64>(q)));
          degs.push_back(blocks[q].size());
        }
        Bond row = Bond(BD_IN, qs, degs, {Symmetry::Zn(n_sectors)});
        UniTensor M({row, row.redirect()}, {"r", "c"}, 1);
        for (unsigned int q = 0; q < n_sectors; q++) {
          Tensor blk = M.get_block_(q);
          for (cytnx_uint64 i = 0; i < blocks[q].size(); i++)
            for (cytnx_uint64 j = 0; j < blocks[q].size(); j++)
              blk.at<double>({i, j}) = blocks[q][i][j];
        }
        return M;
      }

      std::vector<double> block_as_vector(const Tensor &blk) {
        std::vector<double> out(blk.shape()[0]);
        for (cytnx_uint64 i = 0; i < out.size(); i++) out[i] = blk.storage().at<double>(i);
        return out;
      }

      void expect_block_values(const UniTensor &e, cytnx_uint64 b, std::vector<double> expected,
                               double tol) {
        std::vector<double> got = block_as_vector(e.get_block_(b));
        ASSERT_EQ(got.size(), expected.size()) << "block " << b;
        for (cytnx_uint64 i = 0; i < got.size(); i++)
          EXPECT_NEAR(got[i], expected[i], tol) << "block " << b << " entry " << i;
      }

      // M V == V e on the kept eigenpairs: the kept columns are eigenvectors of M.
      void expect_eigen_equation(const UniTensor &M, const UniTensor &V, const UniTensor &e,
                                 double tol) {
        UniTensor Vc = V.relabel("r", "c");
        UniTensor MV = Contract(M, Vc);  // ("r", "_aux_L")
        UniTensor Ve = Contract(V, e).relabel("_aux_R", "_aux_L");  // ("r", "_aux_L")
        EXPECT_TRUE((MV - Ve.permute(MV.labels())).Norm().item() < tol);
      }

      /*=====test info=====
      describe:Z2 block tensor with diagonal blocks, so the eigenvalues and eigenvectors are known
      exactly. The ranking is by |lambda| across both sectors; kept eigenvalues stay ascending.
      ====================*/
      TEST(EighTruncate, Z2KeepsLargestMagnitudesAcrossSectors) {
        const double tol = 1e-12;
        // sector 0: eigenvalues 5, -3, 0.5 ; sector 1: 4, 1
        UniTensor M = hermitian_blocks(2, {{{5, 0, 0}, {0, -3, 0}, {0, 0, 0.5}}, {{4, 0}, {0, 1}}});

        // keepdim 3 by magnitude: 5, 4, -3
        std::vector<UniTensor> out = linalg::Eigh_truncate(M, 3);
        ASSERT_EQ(out.size(), 2u);
        const UniTensor &e = out[0];
        const UniTensor &V = out[1];
        EXPECT_TRUE(e.is_diag());
        EXPECT_EQ(e.bonds()[0].getDegeneracies(), std::vector<cytnx_uint64>({2, 1}));
        expect_block_values(e, 0, {-3, 5}, tol);
        expect_block_values(e, 1, {4}, tol);
        EXPECT_EQ(V.bonds().back().getDegeneracies(), std::vector<cytnx_uint64>({2, 1}));
        expect_unitary(V, "_aux_L", tol);
        expect_eigen_equation(M, V, e, tol);

        // keepdim 1 keeps exactly the largest magnitude; sector 1 disappears
        std::vector<UniTensor> one = linalg::Eigh_truncate(M, 1);
        EXPECT_EQ(one[0].Nblocks(), 1u);
        EXPECT_EQ(one[0].bonds()[0].qnums(), std::vector<std::vector<cytnx_int64>>({{0}}));
        expect_block_values(one[0], 0, {5}, tol);
        EXPECT_EQ(one[1].Nblocks(), 1u);
        expect_eigen_equation(M, one[1], one[0], tol);
      }

      /*=====test info=====
      describe:err drops the eigenpairs with |lambda| below it even when keepdim allows them, and
      return_err reports the discarded magnitudes (largest, or all in descending order).
      ====================*/
      TEST(EighTruncate, ErrAndReturnErr) {
        const double tol = 1e-12;
        UniTensor M = hermitian_blocks(2, {{{5, 0, 0}, {0, -3, 0}, {0, 0, 0.5}}, {{4, 0}, {0, 1}}});

        std::vector<UniTensor> all = linalg::Eigh_truncate(M, 5, 2.0, true, 2);
        ASSERT_EQ(all.size(), 3u);
        expect_block_values(all[0], 0, {-3, 5}, tol);
        expect_block_values(all[0], 1, {4}, tol);
        Tensor dropped = all[2].get_block_();
        ASSERT_EQ(dropped.shape(), std::vector<cytnx_uint64>({2}));
        EXPECT_NEAR(dropped.storage().at<double>(0), 1.0, tol);
        EXPECT_NEAR(dropped.storage().at<double>(1), 0.5, tol);

        std::vector<UniTensor> largest = linalg::Eigh_truncate(M, 5, 2.0, true, 1);
        ASSERT_TRUE(largest[2].get_block_().is_scalar());
        EXPECT_NEAR(largest[2].get_block_().storage().at<double>(0), 1.0, tol);

        // nothing discarded: a scalar zero for return_err = 1, an empty vector for 2
        std::vector<UniTensor> none = linalg::Eigh_truncate(M, 5, 0., true, 1);
        EXPECT_NEAR(none[2].get_block_().storage().at<double>(0), 0.0, tol);
        std::vector<UniTensor> none2 = linalg::Eigh_truncate(M, 5, 0., true, 2);
        EXPECT_EQ(none2[2].get_block_().shape(), std::vector<cytnx_uint64>({0}));

        // is_V = false returns the eigenvalues only (plus the error when requested)
        std::vector<UniTensor> values_only = linalg::Eigh_truncate(M, 2, 0., false, 0);
        ASSERT_EQ(values_only.size(), 1u);
        expect_block_values(values_only[0], 0, {5}, tol);
        expect_block_values(values_only[0], 1, {4}, tol);
      }

      /*=====test info=====
      describe:min_blockdim has the Svd_truncate semantics: positional over the eigenvalue
      blocks, a floor clamped to the block dimension, honoured before keepdim; a zero entry
      imposes no floor and lets the block be dropped. The vector must have one entry per block.
      ====================*/
      TEST(EighTruncate, MinBlockdimFloors) {
        const double tol = 1e-12;
        UniTensor M = hermitian_blocks(2, {{{5, 0, 0}, {0, -3, 0}, {0, 0, 0.5}}, {{4, 0}, {0, 1}}});

        // keepdim 2 with floor {0, 2}: sector 1 keeps both of its values, sector 0 is dropped
        std::vector<UniTensor> floored = linalg::Eigh_truncate(M, 2, {0, 2});
        EXPECT_EQ(floored[0].bonds()[0].getDegeneracies(), std::vector<cytnx_uint64>({2}));
        EXPECT_EQ(floored[0].bonds()[0].qnums(), std::vector<std::vector<cytnx_int64>>({{1}}));
        expect_block_values(floored[0], 0, {1, 4}, tol);
        expect_eigen_equation(M, floored[1], floored[0], tol);

        // the floor beats keepdim: keepdim 1 with {1, 1} keeps one per sector
        std::vector<UniTensor> each = linalg::Eigh_truncate(M, 1, {1, 1});
        EXPECT_EQ(each[0].bonds()[0].getDegeneracies(), std::vector<cytnx_uint64>({1, 1}));
        expect_block_values(each[0], 0, {5}, tol);
        expect_block_values(each[0], 1, {4}, tol);

        // a floor above the block dimension is clamped: a single-element floor applies to every
        // block (spelled as a vector: a one-element brace list would bind to the err overload)
        std::vector<UniTensor> clamped = linalg::Eigh_truncate(M, 1, std::vector<cytnx_uint64>{9});
        EXPECT_EQ(clamped[0].bonds()[0].getDegeneracies(), std::vector<cytnx_uint64>({3, 2}));

        // positional: one entry per block of the eigenvalue UniTensor
        EXPECT_THROW(linalg::Eigh_truncate(M, 1, {1, 1, 1}), std::logic_error);
      }

      /*=====test info=====
      describe:Z3 block tensor with non-diagonal blocks of known spectra: sector 0 [[2,1],[1,2]]
      (1, 3), sector 1 [[0,-1],[-1,0]] (-1, 1), sector 2 [[4]] (4). An exact degeneracy of
      |lambda| at the cut (the pair -1, 1) is kept entirely, as in Svd_truncate.
      ====================*/
      TEST(EighTruncate, Z3NonDiagonalBlocksAndDegeneracyAtTheCut) {
        const double tol = 1e-10;
        UniTensor M = hermitian_blocks(3, {{{2, 1}, {1, 2}}, {{0, -1}, {-1, 0}}, {{4}}});

        std::vector<UniTensor> two = linalg::Eigh_truncate(M, 2, 0., true, 2);
        EXPECT_EQ(two[0].bonds()[0].getDegeneracies(), std::vector<cytnx_uint64>({1, 1}));
        EXPECT_EQ(two[0].bonds()[0].qnums(), std::vector<std::vector<cytnx_int64>>({{0}, {2}}));
        expect_block_values(two[0], 0, {3}, tol);
        expect_block_values(two[0], 1, {4}, tol);
        expect_unitary(two[1], "_aux_L", tol);
        expect_eigen_equation(M, two[1], two[0], tol);
        // keepdim 2 cuts between 3 and the |lambda| = 1 triple: three magnitudes 1 discarded
        Tensor dropped = two[2].get_block_();
        ASSERT_EQ(dropped.shape(), std::vector<cytnx_uint64>({3}));
        for (cytnx_uint64 i = 0; i < 3; i++) EXPECT_NEAR(dropped.storage().at<double>(i), 1.0, tol);

        // keepdim 3 and 4 cut inside the |lambda| = 1 triple (1 in sector 0, -1 and 1 in sector
        // 1): the exact degeneracy is kept whole, so all five eigenpairs survive and nothing is
        // reported as discarded
        for (cytnx_uint64 keepdim : {3, 4}) {
          std::vector<UniTensor> inside = linalg::Eigh_truncate(M, keepdim, 0., true, 2);
          EXPECT_EQ(inside[0].bonds()[0].getDegeneracies(), std::vector<cytnx_uint64>({2, 2, 1}))
            << "keepdim " << keepdim;
          expect_block_values(inside[0], 1, {-1, 1}, tol);
          expect_eigen_equation(M, inside[1], inside[0], tol);
          EXPECT_EQ(inside[2].get_block_().shape(), std::vector<cytnx_uint64>({0}))
            << "keepdim " << keepdim;
        }
      }

      /*=====test info=====
      describe:A DenseUniTensor goes through the same selection: a diagonal matrix with entries
      (1, -4, 2) keeps (-4, 2) at keepdim 2, ascending, and V e V^dagger is the projection.
      ====================*/
      TEST(EighTruncate, DenseUniTensor) {
        const double tol = 1e-12;
        Tensor T = zeros({3, 3}, Type.Double);
        T.at<double>({0, 0}) = 1;
        T.at<double>({1, 1}) = -4;
        T.at<double>({2, 2}) = 2;
        UniTensor M(T, false, 1);
        M.set_labels({"r", "c"});
        std::vector<UniTensor> out = linalg::Eigh_truncate(M, 2, 0., true, 2);
        ASSERT_EQ(out.size(), 3u);
        expect_block_values(out[0], 0, {-4, 2}, tol);
        EXPECT_EQ(out[1].shape(), std::vector<cytnx_uint64>({3, 2}));
        Tensor dropped = out[2].get_block_();
        ASSERT_EQ(dropped.shape(), std::vector<cytnx_uint64>({1}));
        EXPECT_NEAR(dropped.storage().at<double>(0), 1.0, tol);
        // the reconstruction is T with the dropped diagonal entry zeroed
        Tensor Vt = out[1].get_block_();
        Tensor rec = linalg::Matmul(linalg::Matmul(Vt, linalg::Diag(out[0].get_block_())),
                                    Vt.permute({1, 0}).contiguous());
        T.at<double>({0, 0}) = 0;
        EXPECT_TRUE((rec - T).Norm().item() < tol);
      }

      TEST(EighTruncate, RefusesBadInput) {
        UniTensor M = hermitian_blocks(2, {{{1}}, {{2}}});
        EXPECT_THROW(linalg::Eigh_truncate(M, 0), std::logic_error);
        EXPECT_THROW(linalg::Eigh_truncate(M, 1, std::vector<cytnx_uint64>{}), std::logic_error);
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
