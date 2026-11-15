#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

#include "cytnx.hpp"
#include "../../linalg_test/eigh_test_tools.h"

// linalg::Eigh and its `algorithm` option (#1214) on a GPU. cuSOLVER offers the
// divide-and-conquer driver only, so "default" and "dc" must pass the checks of the CPU tests
// (closed-form spectra, residual, orthonormality, reconstruction) and "qr" must be refused.
namespace cytnx {
  namespace gpu_test {
    namespace {

      namespace eigh = cytnx::test::eigh;
      using eigh::Matrix;

      const std::vector<std::string> kGpuAlgorithms = {"default", "dc"};
      const std::vector<unsigned int> kDtypes = {Type.Double, Type.Float, Type.ComplexDouble,
                                                 Type.ComplexFloat};

      std::string describe(const std::string &algorithm, unsigned int dtype) {
        return "algorithm '" + algorithm + "', dtype " + Type.getname(dtype);
      }

      UniTensor block_ut(const std::vector<Tensor> &blocks, const Symmetry &symmetry) {
        std::vector<Qs> qs;
        std::vector<cytnx_uint64> degs;
        for (std::size_t q = 0; q < blocks.size(); q++) {
          qs.push_back(Qs(static_cast<cytnx_int64>(q)));
          degs.push_back(blocks[q].shape()[0]);
        }
        Bond row = Bond(BD_IN, qs, degs, {symmetry});
        UniTensor out({row, row.redirect()}, {"r", "c"}, 1, blocks[0].dtype());
        for (std::size_t q = 0; q < blocks.size(); q++) out.put_block(blocks[q], q);
        return out;
      }

      /*=====test info=====
      describe:Tensor on the GPU, both names of the cuSOLVER driver, every floating dtype:
        distinct eigenvalues of both signs, degenerate eigenvalues, and a 1 x 1 matrix.
      ====================*/
      TEST(Eigh, GpuTensorMatchesClosedFormSpectrum) {
        for (unsigned int dtype : kDtypes) {
          const bool cplx = eigh::is_complex_dtype(dtype);
          const std::complex<double> hopping = cplx ? std::polar(1.3, 0.7) : -1.3;
          Matrix one(1);
          one(0, 0) = -2.5;
          const std::vector<Matrix> matrices = {
            eigh::reflected(eigh::tridiagonal(40, -0.25, hopping), cplx ? 0.9 : 0.),
            eigh::rank_one_update(6, -0.75, 0.02, cplx ? 0.9 : 0.), eigh::ring(6, 0.8), one};
          const std::vector<std::vector<double>> spectra = {
            eigh::tridiagonal_spectrum(40, -0.25, hopping),
            eigh::rank_one_update_spectrum(6, -0.75, 0.02),
            eigh::ring_spectrum(6, 0.8),
            {-2.5}};
          for (std::size_t c = 0; c < matrices.size(); c++) {
            const Tensor a = eigh::to_tensor(matrices[c], dtype).to(Device.cuda);
            for (const std::string &algorithm : kGpuAlgorithms) {
              const std::string context =
                "case " + std::to_string(c) + ", " + describe(algorithm, dtype);
              const std::vector<Tensor> out = linalg::Eigh(a, true, false, algorithm);
              ASSERT_EQ(out.size(), 2u) << context;
              EXPECT_EQ(out[0].device(), Device.cuda) << context;
              EXPECT_EQ(out[1].device(), Device.cuda) << context;
              eigh::expect_eigh(matrices[c], out, dtype, spectra[c], eigh::tolerance(dtype),
                                context);

              const std::vector<Tensor> values_only = linalg::Eigh(a, false, false, algorithm);
              ASSERT_EQ(values_only.size(), 1u) << context;
              eigh::expect_eigenvalues(values_only[0], dtype, spectra[c], eigh::tolerance(dtype),
                                       context + ", values only");
            }
          }
        }
      }

      /*=====test info=====
      describe:BlockUniTensor and BlockFermionicUniTensor on the GPU, sector by sector.
      ====================*/
      TEST(Eigh, GpuBlockUniTensor) {
        for (const Symmetry &symmetry : {Symmetry::Zn(2), Symmetry::FermionParity()}) {
          for (unsigned int dtype : kDtypes) {
            const bool cplx = eigh::is_complex_dtype(dtype);
            const std::complex<double> hopping = cplx ? std::polar(0.9, -0.4) : 0.9;
            const std::vector<Matrix> sectors = {
              eigh::rank_one_update(4, -0.5, 0.1, cplx ? 0.6 : 0.),
              eigh::tridiagonal(3, 0.25, hopping)};
            const std::vector<std::vector<double>> spectra = {
              eigh::rank_one_update_spectrum(4, -0.5, 0.1),
              eigh::tridiagonal_spectrum(3, 0.25, hopping)};
            const UniTensor input =
              block_ut({eigh::to_tensor(sectors[0], dtype), eigh::to_tensor(sectors[1], dtype)},
                       symmetry)
                .to(Device.cuda);
            for (const std::string &algorithm : kGpuAlgorithms) {
              const std::vector<UniTensor> out = linalg::Eigh(input, true, false, algorithm);
              ASSERT_EQ(out.size(), 2u);
              ASSERT_EQ(out[0].Nblocks(), 2u);
              ASSERT_EQ(out[1].Nblocks(), 2u);
              for (std::size_t b = 0; b < 2; b++) {
                const std::string context = describe(algorithm, dtype) + ", " +
                                            input.uten_type_str() + ", sector " + std::to_string(b);
                eigh::expect_eigenvalues(out[0].get_block_(b), dtype, spectra[b],
                                         eigh::tolerance(dtype), context);
                eigh::expect_eigenvectors(sectors[b], out[0].get_block_(b), out[1].get_block_(b),
                                          dtype, eigh::tolerance(dtype), context);
              }
            }
          }
        }
      }

      /*=====test info=====
      describe:"qr" is refused on a GPU, on every entry point, instead of running another
        driver; so is an unknown name.
      ====================*/
      TEST(Eigh, GpuRefusesUnavailableAlgorithm) {
        const Tensor a = eigh::to_tensor(eigh::ring(4, 1.), Type.Double);
        const Tensor a_gpu = a.to(Device.cuda);
        const UniTensor dense = UniTensor(a).to(Device.cuda);
        const UniTensor block = block_ut({a, a}, Symmetry::Zn(2)).to(Device.cuda);
        const UniTensor fermionic = block_ut({a, a}, Symmetry::FermionParity()).to(Device.cuda);
        for (const std::string algorithm : {"qr", "no_such_algorithm", ""}) {
          SCOPED_TRACE("algorithm '" + algorithm + "'");
          EXPECT_THROW(linalg::Eigh(a_gpu, true, false, algorithm), cytnx::error);
          EXPECT_THROW(linalg::Eigh(a_gpu, false, false, algorithm), cytnx::error);
          EXPECT_THROW(a_gpu.Eigh(true, false, algorithm), cytnx::error);
          EXPECT_THROW(linalg::Eigh(dense, true, false, algorithm), cytnx::error);
          EXPECT_THROW(linalg::Eigh(block, true, false, algorithm), cytnx::error);
          EXPECT_THROW(linalg::Eigh(fermionic, true, false, algorithm), cytnx::error);
          EXPECT_THROW(linalg::Eigh_truncate(block, 2, 0., true, 0, 1, algorithm), cytnx::error);
        }
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
