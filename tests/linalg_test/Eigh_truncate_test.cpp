#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

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

      // The full dense array of a block-sparse UniTensor, every block in its sector slot.
      Tensor dense_array_of(const UniTensor &ut) {
        std::vector<cytnx_uint64> shape;
        for (const auto &bond : ut.bonds()) shape.push_back(bond.dim());
        Tensor out = zeros(shape, ut.dtype());
        for (std::size_t b = 0; b < ut.Nblocks(); b++) {
          const std::vector<cytnx_uint64> &qidx = ut.get_qindices(b);
          std::vector<Accessor> where;
          for (std::size_t leg = 0; leg < qidx.size(); leg++) {
            const std::vector<cytnx_uint64> &degs = ut.bonds()[leg].getDegeneracies();
            cytnx_int64 offset = 0;
            for (std::size_t q = 0; q < qidx[leg]; q++) offset += degs[q];
            where.push_back(Accessor::range(offset, offset + degs[qidx[leg]]));
          }
          Tensor blk = ut.get_block_(b);
          if (ut.uten_type() == UTenType.BlockFermionic && ut.signflip()[b]) blk = -blk;
          if (ut.is_diag()) blk = linalg::Diag(blk);
          out.set(where, blk);
        }
        return out;
      }

      // Gauge-invariant content of a truncated eigendecomposition given as dense matrices:
      // the ascending kept eigenvalues, the projector V V^T and the reconstruction V e V^T.
      struct TruncationInvariants {
        std::vector<double> eigenvalues;
        Tensor projector;
        Tensor reconstruction;
      };

      // eigen_matrix is the dense k x k diagonal matrix whose column order matches vectors'.
      TruncationInvariants invariants_of(const Tensor &eigen_matrix, const Tensor &vectors) {
        TruncationInvariants out;
        for (cytnx_uint64 i = 0; i < eigen_matrix.shape()[0]; i++)
          out.eigenvalues.push_back(eigen_matrix.at<double>({i, i}));
        std::sort(out.eigenvalues.begin(), out.eigenvalues.end());
        Tensor vt = vectors.permute({1, 0}).contiguous();
        out.projector = linalg::Matmul(vectors, vt);
        out.reconstruction = linalg::Matmul(linalg::Matmul(vectors, eigen_matrix), vt);
        return out;
      }

      /*=====test info=====
      describe:the same Hermitian matrix through the dense path (dense UniTensor -> Eigh_truncate)
      and through the block-sparse path (block UniTensor -> Eigh_truncate -> assembled to dense)
      must agree on every gauge-invariant object for every keepdim. Non-diagonal Z3 blocks with
      eigenvalues of both signs.
      ====================*/
      TEST(EighTruncate, BlockSparsePathEqualsDensePath) {
        const double tol = 1e-12;
        UniTensor M = hermitian_blocks(3, {{{1.0, 2.0, 0.5}, {2.0, -3.0, 1.5}, {0.5, 1.5, 0.25}},
                                           {{-4.0, 1.0}, {1.0, 2.5}},
                                           {{0.75, -2.0}, {-2.0, 3.5}}});
        Tensor dense = dense_array_of(M);
        UniTensor M_dense(dense, false, 1, {"r", "c"});
        for (cytnx_uint64 keepdim : {1u, 3u, 4u, 6u}) {
          std::vector<UniTensor> block_out = linalg::Eigh_truncate(M, keepdim);
          std::vector<UniTensor> dense_out = linalg::Eigh_truncate(M_dense, keepdim);
          ASSERT_EQ(block_out.size(), 2u);
          ASSERT_EQ(dense_out.size(), 2u);

          TruncationInvariants from_blocks =
            invariants_of(dense_array_of(block_out[0]), dense_array_of(block_out[1]));
          TruncationInvariants from_dense =
            invariants_of(linalg::Diag(dense_out[0].get_block_()), dense_out[1].get_block_());

          ASSERT_EQ(from_blocks.eigenvalues.size(), keepdim) << "keepdim " << keepdim;
          ASSERT_EQ(from_dense.eigenvalues.size(), keepdim) << "keepdim " << keepdim;
          for (std::size_t i = 0; i < keepdim; i++)
            EXPECT_NEAR(from_blocks.eigenvalues[i], from_dense.eigenvalues[i], tol)
              << "keepdim " << keepdim << " eigenvalue " << i;
          EXPECT_TRUE((from_blocks.projector - from_dense.projector).Norm().item() < tol)
            << "keepdim " << keepdim;
          EXPECT_TRUE((from_blocks.reconstruction - from_dense.reconstruction).Norm().item() < tol)
            << "keepdim " << keepdim;
        }
      }

      TEST(EighTruncate, UnsignedDimensionsAndDiscardedReports) {
        const auto block_input = hermitian_blocks(2, {{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}, {{0.5}}});
        const UniTensor dense_input(dense_array_of(block_input), false, 1);
        const std::uint64_t signed_max = std::numeric_limits<std::int64_t>::max();
        for (const auto &input : {dense_input, block_input}) {
          for (std::uint64_t dimension :
               {signed_max, signed_max + 1, std::numeric_limits<std::uint64_t>::max()}) {
            for (bool large_minimum : {false, true}) {
              for (unsigned int return_err : {1u, 2u}) {
                SCOPED_TRACE(input.uten_type_str());
                SCOPED_TRACE(dimension);
                SCOPED_TRACE(large_minimum);
                const auto output = linalg::Eigh_truncate(
                  input, large_minimum ? 4 : dimension, large_minimum ? 100. : 0., false,
                  return_err, large_minimum ? dimension : 1);
                ASSERT_EQ(output.size(), 2u);
                EXPECT_EQ(output[0].bonds()[0].dim(), 4u);
                if (input.uten_type() == UTenType.Dense) {
                  expect_block_values(output[0], 0, {0.5, 1, 2, 3}, 1e-12);
                } else {
                  expect_block_values(output[0], 0, {1, 2, 3}, 1e-12);
                  expect_block_values(output[0], 1, {0.5}, 1e-12);
                }
                if (return_err == 1) {
                  EXPECT_DOUBLE_EQ(static_cast<double>(Scalar(output[1].get_block_().item())), 0.);
                } else {
                  EXPECT_EQ(output[1].get_block_().shape(), std::vector<cytnx_uint64>{0});
                }
              }
            }
          }
        }
      }

      TEST(EighTruncate, FermionicSignflipsDtypesAndSectorDeletion) {
        for (unsigned int dtype :
             {Type.Float, Type.Double, Type.ComplexFloat, Type.ComplexDouble}) {
          SCOPED_TRACE(dtype);
          const bool complex_input = dtype == Type.ComplexFloat || dtype == Type.ComplexDouble;
          const bool single_precision = dtype == Type.Float || dtype == Type.ComplexFloat;
          const double tol = single_precision ? 1e-5 : 1e-12;
          const auto input =
            permute_with_signflips(make_rank4_herm_dt(complex_input).astype(dtype));
          const auto original = input.clone();
          const auto original_signs = input.signflip();
          const auto output = linalg::Eigh_truncate(input, 1, 0., true, 1);
          const auto &eigenvalues = output[0];
          const auto &vectors = output[1];
          const double odd_radius =
            std::sqrt(0.25 * 0.25 + 0.9 * 0.9 + (complex_input ? 0.35 * 0.35 : 0.));
          const double even_radius =
            std::sqrt(0.75 * 0.75 + 0.6 * 0.6 + (complex_input ? 0.2 * 0.2 : 0.));
          ASSERT_EQ(eigenvalues.Nblocks(), 1u);
          // The selected real sector inherits Eigh's Float -> Double promotion when a pending
          // sign flip is materialized by unary minus. The selected complex sector has no flips.
          EXPECT_EQ(eigenvalues.dtype(), dtype == Type.ComplexFloat ? Type.Float : Type.Double);
          EXPECT_EQ(vectors.dtype(), dtype == Type.Float ? Type.Double : dtype);
          EXPECT_NEAR(static_cast<double>(Scalar(eigenvalues.get_block_(0).storage().at(0))),
                      1.05 + std::max(odd_radius, even_radius), tol);
          EXPECT_NEAR(static_cast<double>(Scalar(output[2].get_block_().item())),
                      1.05 + std::min(odd_radius, even_radius), tol);
          EXPECT_EQ(eigenvalues.signflip().size(), eigenvalues.Nblocks());
          EXPECT_EQ(vectors.signflip().size(), vectors.Nblocks());
          for (bool sign : eigenvalues.signflip()) EXPECT_FALSE(sign);
          for (bool sign : vectors.signflip()) EXPECT_FALSE(sign);
          expect_unitary(vectors, "_aux_L", tol);
          // Check the independent coefficient-matrix eigen-equation. A fermionic network
          // contraction introduces additional graded signs beyond this matrix convention.
          const auto matrix = dense_array_of(input).reshape({4, 4});
          const auto basis = dense_array_of(vectors).reshape({4, 1});
          const auto diagonal = dense_array_of(eigenvalues);
          const auto residual = linalg::Matmul(matrix, basis) - linalg::Matmul(basis, diagonal);
          EXPECT_LT(static_cast<double>(residual.norm()), tol);
          EXPECT_EQ(input.signflip(), original_signs);
          EXPECT_LT(static_cast<double>((input - original).norm()), tol);
        }
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
