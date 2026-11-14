#include <gtest/gtest.h>

#include <functional>
#include <numeric>
#include <vector>

#include "cytnx.hpp"

// UniTensor::flattened_dimension() and flattened_dimension(bonds): the dimension of the vector
// space of UniTensors with given bonds, computed from the bonds. The expected values below are
// counted by hand from the sector lists in each test.

namespace cytnx {
  namespace {

    // Number of elements in the blocks the UniTensor actually stores, counted from the block
    // shapes. For a tensor that stores exactly the blocks its bonds allow, this equals
    // flattened_dimension().
    cytnx_uint64 StoredElements(const UniTensor& ut) {
      const auto block_size = [](const Tensor& block) {
        cytnx_uint64 size = 1;
        for (const cytnx_uint64 extent : block.shape()) size *= extent;
        return size;
      };
      if (!ut.is_blockform()) return block_size(ut.get_block_());
      cytnx_uint64 total = 0;
      for (const Tensor& block : ut.get_blocks_()) total += block_size(block);
      return total;
    }

    cytnx_uint64 ShapeProduct(const UniTensor& ut) {
      const std::vector<cytnx_uint64> shape = ut.shape();
      return std::accumulate(shape.begin(), shape.end(), cytnx_uint64{1},
                             std::multiplies<cytnx_uint64>());
    }

    TEST(FlattenedDimension, EmptyBondListIsOne) {
      EXPECT_EQ(flattened_dimension(std::vector<Bond>{}), 1u);
    }

    TEST(FlattenedDimension, DenseIsProductOfBondDimensions) {
      const UniTensor ut(zeros({2, 3, 4}));
      EXPECT_EQ(ut.flattened_dimension(), 24u);
      EXPECT_EQ(StoredElements(ut), 24u);
    }

    TEST(FlattenedDimension, DenseDiagonalIsFirstBondDimension) {
      const UniTensor ut(zeros({5}), true);
      ASSERT_TRUE(ut.is_diag());
      EXPECT_EQ(ut.flattened_dimension(), 5u);
      EXPECT_EQ(StoredElements(ut), 5u);
    }

    // Ket bond {q=1: 2 states, q=0: 1 state} with a bra bond {q=1: 1, q=0: 3}: the allowed
    // sector pairs have q1 - q2 = 0, i.e. (1,1) with 2*1 and (0,0) with 1*3 states, total 5.
    // With the second bond a ket instead, q1 + q2 = 0 allows only (0,0), total 3. The product
    // of the shape is 3*4 = 12 in both cases.
    TEST(FlattenedDimension, U1CountsAllowedSectorsAndDependsOnBondDirection) {
      const Bond ket(BD_KET, {{1}, {0}}, {2, 1});
      const Bond bra(BD_BRA, {{1}, {0}}, {1, 3});
      const Bond ket2(BD_KET, {{1}, {0}}, {1, 3});

      const UniTensor ut_bra({ket, bra}, {"a", "b"}, 1);
      EXPECT_EQ(ut_bra.flattened_dimension(), 5u);
      EXPECT_EQ(flattened_dimension({ket, bra}), 5u);
      EXPECT_EQ(StoredElements(ut_bra), 5u);
      EXPECT_EQ(ShapeProduct(ut_bra), 12u);

      const UniTensor ut_ket({ket, ket2}, {"a", "b"}, 1);
      EXPECT_EQ(ut_ket.flattened_dimension(), 3u);
      EXPECT_EQ(flattened_dimension({ket, ket2}), 3u);
      EXPECT_EQ(StoredElements(ut_ket), 3u);
    }

    // Two spin-1/2 kets {+1, -1} and their fused bra {+2: 1, 0: 2, -2: 1}: the allowed triples
    // (+1,+1,+2), (+1,-1,0), (-1,+1,0), (-1,-1,-2) hold 1 + 2 + 2 + 1 = 6 states, against a
    // shape product of 2*2*4 = 16.
    TEST(FlattenedDimension, U1ThreeLegs) {
      const Bond spin(BD_KET, {{1}, {-1}}, {1, 1});
      const Bond fused(BD_BRA, {{2}, {0}, {-2}}, {1, 2, 1});
      const UniTensor ut({spin, spin, fused}, {"a", "b", "c"}, 2);
      EXPECT_EQ(ut.flattened_dimension(), 6u);
      EXPECT_EQ(StoredElements(ut), 6u);
      EXPECT_EQ(ShapeProduct(ut), 16u);
    }

    // U1 x Z2, quantum numbers written (u1, z2):
    //   ket a = {(1,0): 1, (-1,1): 2}, ket b = {(1,1): 1, (-1,0): 1}, bra c = {(0,1): 2, (0,0): 3}.
    // a + b gives (2,1), (0,0), (0,0), (-2,1); only (0,0) matches a sector of c, namely (0,0)
    // with 3 states. Total 1*1*3 + 2*1*3 = 9; shape product 3*2*5 = 30.
    TEST(FlattenedDimension, U1TimesZ2) {
      const std::vector<Symmetry> syms = {Symmetry::U1(), Symmetry::Zn(2)};
      const Bond a(BD_KET, {{1, 0}, {-1, 1}}, {1, 2}, syms);
      const Bond b(BD_KET, {{1, 1}, {-1, 0}}, {1, 1}, syms);
      const Bond c(BD_BRA, {{0, 1}, {0, 0}}, {2, 3}, syms);
      const UniTensor ut({a, b, c}, {"a", "b", "c"}, 2);
      EXPECT_EQ(ut.flattened_dimension(), 9u);
      EXPECT_EQ(StoredElements(ut), 9u);
      EXPECT_EQ(ShapeProduct(ut), 30u);
    }

    // Fermion parity, each bond {even: 2, odd: 2}, two kets and a bra: the allowed parity
    // patterns eee, eoo, oeo, ooe each hold 2*2*2 states, total 32 of the 64 dense elements.
    TEST(FlattenedDimension, FermionParity) {
      const Bond ket(BD_KET, {{0}, {1}}, {2, 2}, {Symmetry::FermionParity()});
      const Bond bra = ket.redirect();
      const UniTensor ut({ket, ket, bra}, {"a", "b", "c"}, 2);
      ASSERT_EQ(ut.uten_type(), UTenType.BlockFermionic);
      EXPECT_EQ(ut.flattened_dimension(), 32u);
      EXPECT_EQ(StoredElements(ut), 32u);
      EXPECT_EQ(ShapeProduct(ut), 64u);
    }

    // Diagonal block UniTensor on {q=1: 2, q=0: 3}: one diagonal block per sector, 2 + 3 = 5.
    TEST(FlattenedDimension, BlockDiagonalIsFirstBondDimension) {
      const Bond ket(BD_KET, {{1}, {0}}, {2, 3});
      const UniTensor ut({ket, ket.redirect()}, {"a", "b"}, 1, Type.Double, Device.cpu, true);
      ASSERT_TRUE(ut.is_diag());
      EXPECT_EQ(ut.flattened_dimension(), 5u);
      EXPECT_EQ(flattened_dimension(ut.bonds(), true), 5u);
      EXPECT_EQ(StoredElements(ut), 5u);
    }

    // The value is a property of the bonds, so it does not change when the stored blocks are
    // zero, permuted, or produced by a contraction.
    TEST(FlattenedDimension, MatchesStoredBlocksAfterPermuteAndContract) {
      const Bond spin(BD_KET, {{1}, {-1}}, {1, 1});
      const Bond fused(BD_BRA, {{2}, {0}, {-2}}, {1, 2, 1});
      UniTensor ut({spin, spin, fused}, {"a", "b", "c"}, 2);
      ut.permute_({2, 0, 1});
      EXPECT_EQ(ut.flattened_dimension(), 6u);
      EXPECT_EQ(StoredElements(ut), 6u);

      // Contract a and b with a tensor on the conjugate bonds; c (bra) and d (ket) remain.
      const UniTensor left({spin, spin, fused}, {"a", "b", "c"}, 2);
      const UniTensor right({spin.redirect(), spin.redirect(), fused.redirect()}, {"a", "b", "d"},
                            2);
      const UniTensor out = Contract(left, right);
      ASSERT_EQ(out.rank(), 2u);
      EXPECT_EQ(out.flattened_dimension(), flattened_dimension(out.bonds()));
      EXPECT_EQ(out.flattened_dimension(), StoredElements(out));
      // The fused bond and its conjugate: sectors (q, q) with degeneracies 1, 2, 1, so 1+4+1.
      EXPECT_EQ(out.flattened_dimension(), 6u);
    }

    TEST(FlattenedDimension, FullContractionToScalarIsOne) {
      const Bond ket(BD_KET, {{1}, {0}}, {2, 1});
      const Bond bra(BD_BRA, {{1}, {0}}, {1, 3});
      const UniTensor ut({ket, bra}, {"a", "b"}, 1);
      const UniTensor conj({ket.redirect(), bra.redirect()}, {"a", "b"}, 1);
      const UniTensor scalar = Contract(ut, conj);
      ASSERT_TRUE(scalar.is_scalar());
      EXPECT_EQ(scalar.flattened_dimension(), 1u);
    }

    TEST(FlattenedDimension, NoAllowedSectorIsZero) {
      const Bond ket(BD_KET, {{1}, {2}}, {2, 1});
      EXPECT_EQ(flattened_dimension({ket}), 0u);
      EXPECT_EQ(flattened_dimension({ket, ket}), 0u);
    }

    TEST(FlattenedDimension, RejectsInconsistentSymmetries) {
      const Bond u1(BD_KET, {{1}, {0}}, {2, 1});
      const Bond z2(BD_BRA, {{1}, {0}}, {2, 1}, {Symmetry::Zn(2)});
      const Bond plain(3);
      EXPECT_THROW(flattened_dimension({u1, z2}), std::logic_error);
      EXPECT_THROW(flattened_dimension({u1, plain}), std::logic_error);
    }

    TEST(FlattenedDimension, RejectsVoidUniTensor) {
      const UniTensor ut;
      EXPECT_THROW(ut.flattened_dimension(), std::logic_error);
    }

  }  // namespace
}  // namespace cytnx
