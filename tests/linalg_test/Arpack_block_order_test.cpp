#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "cytnx.hpp"

// Regression tests for #738: the ARPACK drivers (Lanczos, Arnoldi) flatten a block UniTensor
// into one vector, and LinOp::matvec is free to return its blocks in another order than the
// input stores them. Every operator below is written directly on the sector coefficients, so
// its spectrum is known without any Cytnx linear algebra, and returns its blocks reordered.

namespace cytnx {
  namespace test {
    namespace {

      using Qindices = std::vector<cytnx_uint64>;

      std::size_t count_elements(const UniTensor& ut) {
        std::size_t count = 0;
        for (const Tensor& block : ut.get_blocks_()) count += block.storage().size();
        return count;
      }

      double block_norm(const Tensor& block) {
        return linalg::Norm(block).astype(Type.Double).item<double>();
      }

      // Lanczos and Arnoldi each have a second overload that differs only in argument types, so
      // the arguments are passed with their exact types.
      std::vector<UniTensor> run_lanczos(LinOp* op, const UniTensor& v0, const std::string& which,
                                         cytnx_uint64 k, double cvg_crit = 1e-13) {
        return linalg::Lanczos(op, v0, which, cytnx_uint64{10000}, cvg_crit, k, true,
                               cytnx_int32{0}, false);
      }

      std::vector<UniTensor> run_arnoldi(LinOp* op, const UniTensor& v0, const std::string& which,
                                         cytnx_uint64 k, double cvg_crit = 1e-13) {
        return linalg::Arnoldi(op, v0, which, cytnx_uint64{10000}, cvg_crit, k, true,
                               cytnx_int32{0}, false);
      }

      // Stores the same tensor with its blocks in the opposite order.
      UniTensor reverse_block_order(UniTensor ut) {
        ut.apply_();  // a fermionic tensor keeps one signflip per block; clear them first
        std::reverse(ut.get_blocks_().begin(), ut.get_blocks_().end());
        std::reverse(ut.get_itoi().begin(), ut.get_itoi().end());
        return ut;
      }

      UniTensor drop_sector(UniTensor ut, const Qindices& sector) {
        auto& qindices = ut.get_itoi();
        const auto found = std::find(qindices.begin(), qindices.end(), sector);
        EXPECT_NE(found, qindices.end());
        ut.get_blocks_().erase(ut.get_blocks_().begin() + (found - qindices.begin()));
        qindices.erase(found);
        return ut;
      }

      // Three bonds with two parity sectors each. The four parity-even sectors hold 2, 3, 12 and
      // 2 elements, so a block written at the offset of another sector is misplaced data.
      std::vector<Bond> parity_bonds(const Symmetry& symmetry) {
        return {Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 2}, {symmetry}),
                Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 1}, {symmetry}),
                Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 3}, {symmetry})};
      }

      UniTensor make_parity_vector(const Symmetry& symmetry, unsigned int dtype) {
        UniTensor ut(parity_bonds(symmetry), {"a", "b", "c"}, 3, dtype);
        ut.uniform_(-1.0, 1.0, 7);
        return ut;
      }

      // Sector weights 1 (000), 4 (011), 6 (101) and 7 (110).
      double sector_weight(const Qindices& sector) {
        return 1.0 + 4.0 * sector[0] + 2.0 * sector[1] + 1.0 * sector[2];
      }
      const Qindices kTopSector = {1, 1, 0};
      constexpr double kTopWeight = 7.0;

      // Multiplies every sector by sector_weight (or by 0 for `annihilated_sector`). The
      // eigenvalues are the weights, each with the multiplicity of its sector, and an eigenvector
      // lives in one sector. The result always holds every sector of `space`; `rearrange` then
      // changes how it is stored.
      class SectorScalingOp : public LinOp {
       public:
        SectorScalingOp(const UniTensor& in_space, std::size_t nx,
                        std::function<UniTensor(UniTensor)> in_rearrange,
                        const Qindices& in_annihilated_sector = {})
            : LinOp("mv", nx, in_space.dtype(), in_space.device()),
              space(in_space),
              rearrange(std::move(in_rearrange)),
              annihilated_sector(in_annihilated_sector) {}

        UniTensor matvec(const UniTensor& in) override {
          UniTensor out = space.clone();
          for (Tensor& block : out.get_blocks_()) block.fill(0);
          for (std::size_t b = 0; b < in.get_blocks_().size(); ++b) {
            const Qindices& sector = in.get_itoi()[b];
            const auto found = std::find(out.get_itoi().begin(), out.get_itoi().end(), sector);
            const double weight = sector == annihilated_sector ? 0.0 : sector_weight(sector);
            out.get_blocks_()[found - out.get_itoi().begin()] =
              (in.get_blocks_()[b] * weight).astype(in.dtype());
          }
          return rearrange(out);
        }

       private:
        UniTensor space;
        std::function<UniTensor(UniTensor)> rearrange;
        Qindices annihilated_sector;
      };

      std::complex<double> first_eigenvalue(const std::vector<UniTensor>& eigs) {
        return eigs[0].get_block_().astype(Type.ComplexDouble).at<cytnx_complex128>({0});
      }

      // The eigenvector of the largest weight is normalized and lives in kTopSector only.
      void expect_top_sector_eigenvector(const UniTensor& vec, double tol) {
        ASSERT_EQ(vec.get_blocks_().size(), 4u);
        for (std::size_t b = 0; b < vec.get_blocks_().size(); ++b) {
          const double norm = block_norm(vec.get_blocks_()[b]);
          EXPECT_NEAR(norm, vec.get_itoi()[b] == kTopSector ? 1.0 : 0.0, tol)
            << "block " << b << " of the eigenvector";
        }
      }

      struct ScalingCase {
        std::string name;
        bool fermionic;
        unsigned int dtype;
        bool use_arnoldi;
      };

      class ArpackBlockOrder : public ::testing::TestWithParam<ScalingCase> {};

      /*=====test info=====
      describe: matvec multiplies each sector by a known weight and returns the blocks in
                reversed order. The largest-magnitude eigenvalue must be the largest weight (7)
                and its eigenvector must live in that weight's sector.
      ====================*/
      TEST_P(ArpackBlockOrder, ReversedBlocks) {
        const ScalingCase& param = GetParam();
        const Symmetry symmetry = param.fermionic ? Symmetry::FermionParity() : Symmetry::Zn(2);
        const UniTensor v0 = make_parity_vector(symmetry, param.dtype);
        ASSERT_EQ(v0.uten_type(), param.fermionic ? UTenType.BlockFermionic : UTenType.Block);
        ASSERT_EQ(count_elements(v0), 19u);
        // The reordered tensor is the same vector: block arithmetic matches sectors by qindices.
        ASSERT_EQ(block_norm((reverse_block_order(v0.clone()) - v0).Norm()), 0.0);

        SectorScalingOp op(v0, count_elements(v0), reverse_block_order);
        const bool single = param.dtype == Type.Float || param.dtype == Type.ComplexFloat;
        const double tol = single ? 1e-3 : 1e-9;
        const double cvg_crit = single ? 1e-6 : 1e-13;
        const auto eigs = param.use_arnoldi ? run_arnoldi(&op, v0, "LM", 1, cvg_crit)
                                            : run_lanczos(&op, v0, "LM", 1, cvg_crit);
        EXPECT_NEAR(first_eigenvalue(eigs).real(), kTopWeight, tol);
        EXPECT_NEAR(first_eigenvalue(eigs).imag(), 0.0, tol);
        expect_top_sector_eigenvector(eigs[1], single ? 1e-2 : 1e-6);
      }

      INSTANTIATE_TEST_SUITE_P(
        Solvers, ArpackBlockOrder,
        ::testing::Values(
          ScalingCase{"LanczosBlockDouble", false, Type.Double, false},
          ScalingCase{"LanczosBlockFloat", false, Type.Float, false},
          ScalingCase{"LanczosFermionicDouble", true, Type.Double, false},
          ScalingCase{"LanczosBlockComplexDouble", false, Type.ComplexDouble, false},
          ScalingCase{"ArnoldiBlockDouble", false, Type.Double, true},
          ScalingCase{"ArnoldiBlockComplexDouble", false, Type.ComplexDouble, true},
          ScalingCase{"ArnoldiBlockComplexFloat", false, Type.ComplexFloat, true},
          ScalingCase{"ArnoldiFermionicDouble", true, Type.Double, true},
          ScalingCase{"ArnoldiFermionicComplexDouble", true, Type.ComplexDouble, true}),
        [](const ::testing::TestParamInfo<ScalingCase>& info) { return info.param.name; });

      /*=====test info=====
      describe: matvec annihilates one sector and returns a tensor that does not store that
                block at all (and stores the others in reversed order). A sector that is absent
                from the output is a zero block, so the result is the same as above.
      ====================*/
      TEST(ArpackBlockOrder, MissingBlockIsZeroSector) {
        const Qindices annihilated = {0, 1, 1};
        const auto drop_and_reverse = [&](UniTensor ut) {
          return reverse_block_order(drop_sector(ut, annihilated));
        };
        for (const unsigned int dtype : {Type.Double, Type.ComplexDouble}) {
          const UniTensor v0 = make_parity_vector(Symmetry::Zn(2), dtype);
          SectorScalingOp op(v0, count_elements(v0), drop_and_reverse, annihilated);
          const auto eigs = run_lanczos(&op, v0, "LM", 1);
          EXPECT_NEAR(first_eigenvalue(eigs).real(), kTopWeight, 1e-9);
          EXPECT_NEAR(first_eigenvalue(eigs).imag(), 0.0, 1e-9);
          expect_top_sector_eigenvector(eigs[1], 1e-6);
        }
      }

      /*=====test info=====
      describe: the start vector does not store one sector, but matvec returns a block in it.
                The output is outside the space the iteration runs in, which must be an error
                rather than a write past the ARPACK buffer.
      ====================*/
      TEST(ArpackBlockOrder, ExtraBlockIsAnError) {
        const auto keep = [](UniTensor ut) { return ut; };
        for (const unsigned int dtype : {Type.Double, Type.ComplexDouble}) {
          const UniTensor space = make_parity_vector(Symmetry::Zn(2), dtype);
          const UniTensor v0 = drop_sector(space.clone(), {0, 0, 0});
          ASSERT_EQ(count_elements(v0), 17u);
          SectorScalingOp op(space, count_elements(v0), keep);
          EXPECT_THROW(run_lanczos(&op, v0, "LM", 1), std::logic_error);
          if (dtype == Type.Double) {
            EXPECT_THROW(run_arnoldi(&op, v0, "LM", 1), std::logic_error);
          }
        }
      }

      /*=====test info=====
      describe: matvec returns a tensor with the same number of elements but with the sectors of
                one bond listed in the opposite order. Its qindices then name other sectors, so
                it cannot be mapped onto the input and must be rejected.
      ====================*/
      TEST(ArpackBlockOrder, DifferentBondsAreAnError) {
        class OtherBondsOp : public LinOp {
         public:
          explicit OtherBondsOp(std::size_t nx) : LinOp("mv", nx, Type.Double) {}
          UniTensor matvec(const UniTensor& in) override {
            std::vector<Bond> bonds = in.bonds();
            bonds[1] = Bond(BD_IN, {Qs(1) >> 1, Qs(0) >> 2}, {Symmetry::Zn(2)});
            UniTensor out(bonds, in.labels(), in.rowrank(), in.dtype());
            out.uniform_(-1.0, 1.0, 3);
            return out;
          }
        };
        const UniTensor v0 = make_parity_vector(Symmetry::Zn(2), Type.Double);
        OtherBondsOp op(count_elements(v0));
        EXPECT_THROW(run_lanczos(&op, v0, "LM", 1), std::logic_error);
        EXPECT_THROW(run_arnoldi(&op, v0, "LM", 1), std::logic_error);
      }

      // Spin-1/2 Heisenberg chain H = sum_i S_i . S_{i+1} on four sites with open ends, acting
      // on the coefficients of the Sz_total = 0 basis states. A basis state is the qindices of
      // its 1x1 block: 0 is spin up, 1 is spin down.
      std::map<Qindices, double> apply_heisenberg(const std::map<Qindices, double>& psi) {
        std::map<Qindices, double> out;
        for (const auto& [state, coefficient] : psi) out[state] = 0.0;
        for (const auto& [state, coefficient] : psi) {
          for (std::size_t site = 0; site + 1 < state.size(); ++site) {
            if (state[site] == state[site + 1]) {
              out[state] += 0.25 * coefficient;
            } else {
              out[state] -= 0.25 * coefficient;
              Qindices flipped = state;
              std::swap(flipped[site], flipped[site + 1]);
              out[flipped] += 0.5 * coefficient;
            }
          }
        }
        return out;
      }

      std::map<Qindices, double> to_coefficients(const UniTensor& ut) {
        std::map<Qindices, double> psi;
        for (std::size_t b = 0; b < ut.get_blocks_().size(); ++b) {
          psi[ut.get_itoi()[b]] = ut.get_blocks_()[b].at<double>({0, 0, 0, 0});
        }
        return psi;
      }

      class HeisenbergOp : public LinOp {
       public:
        explicit HeisenbergOp(std::size_t nx) : LinOp("mv", nx, Type.Double) {}
        UniTensor matvec(const UniTensor& in) override {
          const auto h_psi = apply_heisenberg(to_coefficients(in));
          UniTensor out = reverse_block_order(in.clone());
          for (std::size_t b = 0; b < out.get_blocks_().size(); ++b) {
            out.get_blocks_()[b].at<double>({0, 0, 0, 0}) = h_psi.at(out.get_itoi()[b]);
          }
          return out;
        }
      };

      /*=====test info=====
      describe: an operator that couples the blocks. The Sz_total = 0 sector of the four-site
                open Heisenberg chain has six 1x1 blocks; its two lowest energies are
                -(3 + 2 sqrt(3)) / 4 (total spin 0) and -(1 + 2 sqrt(2)) / 4 (total spin 1).
                matvec returns the blocks in reversed order. Checks both energies and the
                residual H v - E v of both eigenvectors.
      ====================*/
      TEST(ArpackBlockOrder, HeisenbergChainReversedBlocks) {
        const Bond spin = Bond(BD_IN, {Qs(1) >> 1, Qs(-1) >> 1}, {Symmetry::U1()});
        UniTensor v0({spin, spin, spin, spin}, {"s0", "s1", "s2", "s3"}, 4, Type.Double);
        ASSERT_EQ(v0.get_blocks_().size(), 6u);
        v0.uniform_(-1.0, 1.0, 11);

        const std::vector<double> expected = {-(3.0 + 2.0 * std::sqrt(3.0)) / 4.0,
                                              -(1.0 + 2.0 * std::sqrt(2.0)) / 4.0};
        HeisenbergOp op(count_elements(v0));
        const auto eigs = run_lanczos(&op, v0, "SA", 2);
        ASSERT_EQ(eigs.size(), 3u);
        for (std::size_t k = 0; k < 2; ++k) {
          const double energy = eigs[0].get_block_().at<double>({k});
          EXPECT_NEAR(energy, expected[k], 1e-10);
          const auto psi = to_coefficients(eigs[k + 1]);
          const auto h_psi = apply_heisenberg(psi);
          double norm2 = 0.0;
          for (const auto& [state, coefficient] : psi) {
            EXPECT_NEAR(h_psi.at(state), energy * coefficient, 1e-9);
            norm2 += coefficient * coefficient;
          }
          EXPECT_NEAR(norm2, 1.0, 1e-10);
        }
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
