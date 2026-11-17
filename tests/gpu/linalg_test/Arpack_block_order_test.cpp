#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <string>
#include <vector>

#include "cytnx.hpp"

// GPU counterpart of the #738 regression test in tests/linalg_test/Arpack_block_order_test.cpp:
// LinOp::matvec returns the blocks of a block UniTensor on the GPU in another order than the
// input stores them. The operator multiplies each sector by a known weight, so its spectrum is
// known without any Cytnx linear algebra.

namespace cytnx {
  namespace gpu_test {
    namespace {

      using Qindices = std::vector<cytnx_uint64>;

      // Sector weights 1 (000), 4 (011), 6 (101) and 7 (110).
      double sector_weight(const Qindices& sector) {
        return 1.0 + 4.0 * sector[0] + 2.0 * sector[1] + 1.0 * sector[2];
      }
      const Qindices kTopSector = {1, 1, 0};
      constexpr double kTopWeight = 7.0;

      // The four parity-even sectors hold 2, 3, 12 and 2 elements.
      UniTensor make_parity_vector(const Symmetry& symmetry, unsigned int dtype) {
        const std::vector<Bond> bonds = {Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 2}, {symmetry}),
                                         Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 1}, {symmetry}),
                                         Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 3}, {symmetry})};
        UniTensor ut(bonds, {"a", "b", "c"}, 3, dtype);
        ut.uniform_(-1.0, 1.0, 7);
        return ut.to(Device.cuda);
      }

      class GpuSectorScalingOp : public LinOp {
       public:
        GpuSectorScalingOp(std::size_t nx, unsigned int dtype)
            : LinOp("mv", nx, dtype, Device.cuda) {}

        UniTensor matvec(const UniTensor& in) override {
          UniTensor out = in.clone();
          out.apply_();  // a fermionic tensor keeps one signflip per block; clear them first
          for (std::size_t b = 0; b < out.get_blocks_().size(); ++b) {
            out.get_blocks_()[b] =
              (in.get_blocks_()[b] * sector_weight(in.get_itoi()[b])).astype(in.dtype());
          }
          std::reverse(out.get_blocks_().begin(), out.get_blocks_().end());
          std::reverse(out.get_itoi().begin(), out.get_itoi().end());
          return out;
        }
      };

      struct ScalingCase {
        std::string name;
        bool fermionic;
        unsigned int dtype;
        bool use_arnoldi;
      };

      class GpuArpackBlockOrder : public ::testing::TestWithParam<ScalingCase> {};

      /*=====test info=====
      describe: matvec multiplies each sector by a known weight and returns the blocks in
                reversed order, all on the GPU. The largest-magnitude eigenvalue must be the
                largest weight (7) and its eigenvector must live in that weight's sector.
      ====================*/
      TEST_P(GpuArpackBlockOrder, ReversedBlocks) {
        const ScalingCase& param = GetParam();
        const Symmetry symmetry = param.fermionic ? Symmetry::FermionParity() : Symmetry::Zn(2);
        const UniTensor v0 = make_parity_vector(symmetry, param.dtype);
        ASSERT_EQ(v0.device(), Device.cuda);
        ASSERT_EQ(v0.get_blocks_().size(), 4u);

        GpuSectorScalingOp op(19, param.dtype);
        // Lanczos and Arnoldi each have a second overload that differs only in argument types, so
        // the arguments are passed with their exact types.
        const auto eigs = param.use_arnoldi
                            ? linalg::Arnoldi(&op, v0, "LM", cytnx_uint64{10000}, 1e-13,
                                              cytnx_uint64{1}, true, cytnx_int32{0}, false)
                            : linalg::Lanczos(&op, v0, "LM", cytnx_uint64{10000}, 1e-13,
                                              cytnx_uint64{1}, true, cytnx_int32{0}, false);
        ASSERT_EQ(eigs.size(), 2u);
        EXPECT_EQ(eigs[1].device(), Device.cuda);

        const std::complex<double> eigenvalue =
          eigs[0].to(Device.cpu).get_block_().astype(Type.ComplexDouble).at<cytnx_complex128>({0});
        EXPECT_NEAR(eigenvalue.real(), kTopWeight, 1e-9);
        EXPECT_NEAR(eigenvalue.imag(), 0.0, 1e-9);

        const UniTensor vec = eigs[1].to(Device.cpu);
        ASSERT_EQ(vec.get_blocks_().size(), 4u);
        for (std::size_t b = 0; b < vec.get_blocks_().size(); ++b) {
          const double norm = linalg::Norm(vec.get_blocks_()[b]).astype(Type.Double).item<double>();
          EXPECT_NEAR(norm, vec.get_itoi()[b] == kTopSector ? 1.0 : 0.0, 1e-6)
            << "block " << b << " of the eigenvector";
        }
      }

      INSTANTIATE_TEST_SUITE_P(
        Solvers, GpuArpackBlockOrder,
        ::testing::Values(ScalingCase{"LanczosBlockDouble", false, Type.Double, false},
                          ScalingCase{"LanczosFermionicDouble", true, Type.Double, false},
                          ScalingCase{"ArnoldiBlockDouble", false, Type.Double, true},
                          ScalingCase{"ArnoldiBlockComplexDouble", false, Type.ComplexDouble, true},
                          ScalingCase{"ArnoldiFermionicComplexDouble", true, Type.ComplexDouble,
                                      true}),
        [](const ::testing::TestParamInfo<ScalingCase>& info) { return info.param.name; });

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
