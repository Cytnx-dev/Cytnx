#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

#include "cytnx.hpp"
#include "eigh_test_tools.h"

// linalg::Eigh and its `algorithm` option (#1214). Every expectation is a closed-form spectrum
// or a property of the decomposition (residual, orthonormality, reconstruction) evaluated in
// eigh_test_tools.h; no test pins an eigenvector sign or a basis of a degenerate eigenspace.
namespace cytnx {
  namespace test {
    namespace {

      using eigh::Matrix;

      // The drivers that the CPU offers; each has to pass the same checks.
      const std::vector<std::string> kCpuAlgorithms = {"default", "qr", "dc"};
      const std::vector<unsigned int> kDtypes = {Type.Double, Type.Float, Type.ComplexDouble,
                                                 Type.ComplexFloat};

      std::string describe(const std::string &algorithm, unsigned int dtype) {
        return "algorithm '" + algorithm + "', dtype " + Type.getname(dtype);
      }

      // A Hermitian matrix of the dtype with a closed-form spectrum.
      struct Case {
        std::string name;
        Matrix matrix;
        std::vector<double> spectrum;
      };

      // The cases for a dtype: complex dtypes get matrices that are complex.
      std::vector<Case> cases_for(unsigned int dtype) {
        const bool cplx = eigh::is_complex_dtype(dtype);
        const std::complex<double> hopping = cplx ? std::polar(1.3, 0.7) : -1.3;
        const double phase = cplx ? 0.9 : 0.;
        std::vector<Case> cases;
        // distinct eigenvalues of both signs
        cases.push_back({"tridiagonal 7", eigh::tridiagonal(7, 0.5, hopping),
                         eigh::tridiagonal_spectrum(7, 0.5, hopping)});
        // above the size (25) at which LAPACK's divide and conquer starts to divide
        cases.push_back({"tridiagonal 40", eigh::tridiagonal(40, -0.25, hopping),
                         eigh::tridiagonal_spectrum(40, -0.25, hopping)});
        // a dense matrix with distinct eigenvalues, also above that size
        cases.push_back({"dense 33", eigh::reflected(eigh::tridiagonal(33, 0.4, hopping), phase),
                         eigh::tridiagonal_spectrum(33, 0.4, hopping)});
        // a dense matrix with a 29-fold degenerate eigenvalue
        cases.push_back({"rank-one update 30", eigh::rank_one_update(30, -0.75, 1e-3, phase),
                         eigh::rank_one_update_spectrum(30, -0.75, 1e-3)});
        // a negative, five-fold degenerate eigenvalue and one positive eigenvalue
        cases.push_back({"rank-one update 6", eigh::rank_one_update(6, -0.75, 0.02, phase),
                         eigh::rank_one_update_spectrum(6, -0.75, 0.02)});
        // several two-fold degenerate eigenvalues of both signs
        cases.push_back({"ring 6", eigh::ring(6, 0.8), eigh::ring_spectrum(6, 0.8)});
        // a zero matrix: one n-fold degenerate eigenvalue
        cases.push_back({"zero 4", Matrix(4), std::vector<double>(4, 0.)});
        // 1 x 1 with a negative eigenvalue
        Matrix one(1);
        one(0, 0) = -2.5;
        cases.push_back({"1 x 1", one, {-2.5}});
        return cases;
      }

      // A rank-2 symmetric UniTensor with one block per charge sector of `symmetry`.
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
      describe:Tensor, every CPU driver, every floating dtype: eigenvalues against the
        closed-form spectrum, residual, orthonormality and reconstruction. The cases include
        degenerate spectra, negative eigenvalues, a zero matrix and a 1 x 1 matrix.
      ====================*/
      TEST(Eigh, TensorMatchesClosedFormSpectrum) {
        for (unsigned int dtype : kDtypes) {
          for (const Case &c : cases_for(dtype)) {
            const Tensor a = eigh::to_tensor(c.matrix, dtype);
            for (const std::string &algorithm : kCpuAlgorithms) {
              const std::string context = c.name + ", " + describe(algorithm, dtype);
              eigh::expect_eigh(c.matrix, linalg::Eigh(a, true, false, algorithm), dtype,
                                c.spectrum, eigh::tolerance(dtype), context);
            }
          }
        }
      }

      /*=====test info=====
      describe:is_V = false returns the eigenvalues only, for every driver and dtype.
      ====================*/
      TEST(Eigh, TensorEigenvaluesOnly) {
        for (unsigned int dtype : kDtypes) {
          for (const Case &c : cases_for(dtype)) {
            const Tensor a = eigh::to_tensor(c.matrix, dtype);
            for (const std::string &algorithm : kCpuAlgorithms) {
              const std::vector<Tensor> out = linalg::Eigh(a, false, false, algorithm);
              ASSERT_EQ(out.size(), 1u);
              eigh::expect_eigenvalues(out[0], dtype, c.spectrum, eigh::tolerance(dtype),
                                       c.name + ", " + describe(algorithm, dtype));
            }
          }
        }
      }

      /*=====test info=====
      describe:row_v = true returns V^dagger: its conjugate transpose passes the eigenvector
        checks.
      ====================*/
      TEST(Eigh, TensorRowVectors) {
        for (unsigned int dtype : kDtypes) {
          const Case c = cases_for(dtype)[0];
          const Tensor a = eigh::to_tensor(c.matrix, dtype);
          for (const std::string &algorithm : kCpuAlgorithms) {
            const std::vector<Tensor> out = linalg::Eigh(a, true, true, algorithm);
            ASSERT_EQ(out.size(), 2u);
            const Matrix rows = eigh::to_matrix(out[1]);
            Matrix columns(rows.n);
            for (std::size_t i = 0; i < rows.n; i++)
              for (std::size_t j = 0; j < rows.n; j++) columns(i, j) = std::conj(rows(j, i));
            eigh::expect_eigenvectors(
              c.matrix, out[0], eigh::to_tensor(columns, Type.ComplexDouble), Type.ComplexDouble,
              eigh::tolerance(dtype), describe(algorithm, dtype));
          }
        }
      }

      /*=====test info=====
      describe:The Tensor member function forwards the option.
      ====================*/
      TEST(Eigh, TensorMemberFunction) {
        const Case c = cases_for(Type.Double)[0];
        const Tensor a = eigh::to_tensor(c.matrix, Type.Double);
        for (const std::string &algorithm : kCpuAlgorithms) {
          eigh::expect_eigh(c.matrix, a.Eigh(true, false, algorithm), Type.Double, c.spectrum,
                            eigh::tolerance(Type.Double), describe(algorithm, Type.Double));
        }
        EXPECT_THROW(a.Eigh(true, false, "no_such_algorithm"), cytnx::error);
      }

      /*=====test info=====
      describe:An integer matrix is decomposed as Double by every driver. diag(3, -4) plus the
        off-diagonal 2 has the eigenvalues (-1 +- sqrt(65)) / 2.
      ====================*/
      TEST(Eigh, IntegerInputIsDecomposedAsDouble) {
        Tensor a({2, 2}, Type.Int64);
        a.at<cytnx_int64>({0, 0}) = 3;
        a.at<cytnx_int64>({0, 1}) = 2;
        a.at<cytnx_int64>({1, 0}) = 2;
        a.at<cytnx_int64>({1, 1}) = -4;
        Matrix m(2);
        m(0, 0) = 3;
        m(0, 1) = m(1, 0) = 2;
        m(1, 1) = -4;
        const std::vector<double> spectrum = {(-1. - std::sqrt(65.)) / 2.,
                                              (-1. + std::sqrt(65.)) / 2.};
        for (const std::string &algorithm : kCpuAlgorithms) {
          eigh::expect_eigh(m, linalg::Eigh(a, true, false, algorithm), Type.Double, spectrum,
                            1e-12, "algorithm '" + algorithm + "'");
        }
      }

      /*=====test info=====
      describe:A name that is not an algorithm raises cytnx::error on every entry point, also
        when no eigenvectors are requested. Names are case sensitive, and the names of drivers
        that are not implemented are refused like any other unknown name.
      ====================*/
      TEST(Eigh, UnknownAlgorithmRaises) {
        const Tensor a = eigh::to_tensor(eigh::ring(4, 1.), Type.Double);
        const UniTensor dense(a);
        const UniTensor block = block_ut({a, a}, Symmetry::Zn(2));
        const UniTensor fermionic = block_ut({a, a}, Symmetry::FermionParity());
        ASSERT_EQ(block.uten_type(), UTenType.Block);
        ASSERT_EQ(fermionic.uten_type(), UTenType.BlockFermionic);
        for (const std::string algorithm :
             {"no_such_algorithm", "", "DC", "Qr", "mrrr", "jacobi"}) {
          SCOPED_TRACE("algorithm '" + algorithm + "'");
          EXPECT_THROW(linalg::Eigh(a, true, false, algorithm), cytnx::error);
          EXPECT_THROW(linalg::Eigh(a, false, false, algorithm), cytnx::error);
          EXPECT_THROW(linalg::Eigh(dense, true, false, algorithm), cytnx::error);
          EXPECT_THROW(linalg::Eigh(block, true, false, algorithm), cytnx::error);
          EXPECT_THROW(linalg::Eigh(fermionic, true, false, algorithm), cytnx::error);
          EXPECT_THROW(linalg::Eigh_truncate(dense, 2, 0., true, 0, 1, algorithm), cytnx::error);
          EXPECT_THROW(linalg::Eigh_truncate(block, 2, 0., true, 0, 1, algorithm), cytnx::error);
          EXPECT_THROW(linalg::Eigh_truncate(block, 2, {1}, 0., true, 0, 1, algorithm),
                       cytnx::error);
        }
      }

      /*=====test info=====
      describe:DenseUniTensor, every driver and dtype. A rank-4 tensor with rowrank 2 is
        decomposed as the matrix of its row and column spaces.
      ====================*/
      TEST(Eigh, DenseUniTensor) {
        for (unsigned int dtype : kDtypes) {
          const Case c = cases_for(dtype)[4];  // rank-one update 6: a degenerate eigenvalue
          const Tensor a = eigh::to_tensor(c.matrix, dtype);
          const UniTensor rank2(a);
          UniTensor rank4(a.reshape({2, 3, 2, 3}), false, 2);
          for (const std::string &algorithm : kCpuAlgorithms) {
            for (const UniTensor &input : {rank2, rank4}) {
              const std::string context =
                describe(algorithm, dtype) + ", rank " + std::to_string(input.rank());
              const std::vector<UniTensor> out = linalg::Eigh(input, true, false, algorithm);
              ASSERT_EQ(out.size(), 2u) << context;
              EXPECT_TRUE(out[0].is_diag()) << context;
              eigh::expect_eigenvalues(out[0].get_block_(), dtype, c.spectrum,
                                       eigh::tolerance(dtype), context);
              eigh::expect_eigenvectors(c.matrix, out[0].get_block_(), out[1].get_block_(), dtype,
                                        eigh::tolerance(dtype), context);
            }
          }
        }
      }

      // The checks shared by the two block types, on the first `n_sectors` of three sectors: a
      // degenerate spectrum, a spectrum of distinct eigenvalues and a 1 x 1 block.
      void expect_block_eigh(const Symmetry &symmetry, int uten_type, std::size_t n_sectors) {
        for (unsigned int dtype : kDtypes) {
          const bool cplx = eigh::is_complex_dtype(dtype);
          const std::complex<double> hopping = cplx ? std::polar(0.9, -0.4) : 0.9;
          Matrix one(1);
          one(0, 0) = -1.75;
          const std::vector<Matrix> sectors = {eigh::rank_one_update(4, -0.5, 0.1, cplx ? 0.6 : 0.),
                                               eigh::tridiagonal(3, 0.25, hopping), one};
          const std::vector<std::vector<double>> spectra = {
            eigh::rank_one_update_spectrum(4, -0.5, 0.1),
            eigh::tridiagonal_spectrum(3, 0.25, hopping),
            {-1.75}};
          std::vector<Tensor> blocks;
          for (std::size_t q = 0; q < n_sectors; q++)
            blocks.push_back(eigh::to_tensor(sectors[q], dtype));
          const UniTensor input = block_ut(blocks, symmetry);
          ASSERT_EQ(input.uten_type(), uten_type);

          for (const std::string &algorithm : kCpuAlgorithms) {
            const std::string context = describe(algorithm, dtype);
            const std::vector<UniTensor> out = linalg::Eigh(input, true, false, algorithm);
            ASSERT_EQ(out.size(), 2u) << context;
            const UniTensor &e = out[0];
            const UniTensor &v = out[1];
            EXPECT_EQ(e.uten_type(), uten_type) << context;
            EXPECT_EQ(v.uten_type(), uten_type) << context;
            EXPECT_TRUE(e.is_diag()) << context;
            ASSERT_EQ(e.Nblocks(), blocks.size()) << context;
            ASSERT_EQ(v.Nblocks(), blocks.size()) << context;
            for (std::size_t b = 0; b < blocks.size(); b++) {
              const std::string sector = context + ", sector " + std::to_string(b);
              eigh::expect_eigenvalues(e.get_block_(b), dtype, spectra[b], eigh::tolerance(dtype),
                                       sector);
              eigh::expect_eigenvectors(sectors[b], e.get_block_(b), v.get_block_(b), dtype,
                                        eigh::tolerance(dtype), sector);
            }

            const std::vector<UniTensor> values_only = linalg::Eigh(input, false, false, algorithm);
            ASSERT_EQ(values_only.size(), 1u) << context;
            for (std::size_t b = 0; b < blocks.size(); b++) {
              eigh::expect_eigenvalues(values_only[0].get_block_(b), dtype, spectra[b],
                                       eigh::tolerance(dtype),
                                       context + ", values only, sector " + std::to_string(b));
            }
          }
        }
      }

      /*=====test info=====
      describe:BlockUniTensor with three Z3 sectors, every driver and dtype, sector by sector.
      ====================*/
      TEST(Eigh, BlockUniTensor) { expect_block_eigh(Symmetry::Zn(3), UTenType.Block, 3); }

      /*=====test info=====
      describe:BlockFermionicUniTensor with the two fermion-parity sectors, every driver and
        dtype, sector by sector.
      ====================*/
      TEST(Eigh, BlockFermionicUniTensor) {
        expect_block_eigh(Symmetry::FermionParity(), UTenType.BlockFermionic, 2);
      }

      /*=====test info=====
      describe:Eigh_truncate forwards the option. Sector 0 has the eigenvalues 5, -3, 0.5 and
        sector 1 has 4, 1; the three of largest magnitude are 5, 4, -3.
      ====================*/
      TEST(Eigh, TruncateForwardsAlgorithm) {
        Matrix first(3), second(2);
        first(0, 0) = 5;
        first(1, 1) = -3;
        first(2, 2) = 0.5;
        second(0, 0) = 4;
        second(1, 1) = 1;
        const UniTensor input =
          block_ut({eigh::to_tensor(first, Type.Double), eigh::to_tensor(second, Type.Double)},
                   Symmetry::Zn(2));
        for (const std::string &algorithm : kCpuAlgorithms) {
          const std::string context = "algorithm '" + algorithm + "'";
          for (const std::vector<UniTensor> &out :
               {linalg::Eigh_truncate(input, 3, 0., true, 0, 1, algorithm),
                linalg::Eigh_truncate(input, 3, {1}, 0., true, 0, 1, algorithm)}) {
            ASSERT_EQ(out.size(), 2u) << context;
            eigh::expect_eigenvalues(out[0].get_block_(0), Type.Double, {-3, 5}, 1e-12, context);
            eigh::expect_eigenvalues(out[0].get_block_(1), Type.Double, {4}, 1e-12, context);
          }
        }
      }

      /*=====test info=====
      describe:The UniTensor paths and Eigh_truncate hand the option on to the per-sector
        decomposition: for a named driver they return, bit for bit, what Eigh returns for the
        sector's matrix as a Tensor with that driver. The value checks above cannot see a
        dropped option, because every driver passes them. The sector is large enough for the
        two drivers to differ in their rounding.
      ====================*/
      TEST(Eigh, UniTensorPathsForwardAlgorithm) {
        const Case c = cases_for(Type.Double)[2];  // dense 33
        const Tensor a = eigh::to_tensor(c.matrix, Type.Double);
        const std::vector<UniTensor> inputs = {UniTensor(a), block_ut({a, a}, Symmetry::Zn(2)),
                                               block_ut({a, a}, Symmetry::FermionParity())};
        for (const std::string algorithm : {"qr", "dc"}) {
          const std::vector<Tensor> expected = linalg::Eigh(a, true, false, algorithm);
          const std::vector<double> expected_e = eigh::to_values(expected[0]);
          const auto expected_v = eigh::to_matrix(expected[1]).data;
          for (const UniTensor &input : inputs) {
            const std::string context =
              "algorithm '" + algorithm + "', " + input.uten_type_str() + " UniTensor";
            const cytnx_uint64 full_dim = input.bonds()[0].dim();
            for (const std::vector<UniTensor> &out :
                 {linalg::Eigh(input, true, false, algorithm),
                  linalg::Eigh_truncate(input, full_dim, 0., true, 0, 1, algorithm),
                  linalg::Eigh_truncate(input, full_dim, {1}, 0., true, 0, 1, algorithm)}) {
              ASSERT_EQ(out.size(), 2u) << context;
              for (cytnx_uint64 b = 0; b < out[0].Nblocks(); b++) {
                EXPECT_EQ(eigh::to_values(out[0].get_block_(b)), expected_e) << context;
                EXPECT_EQ(eigh::to_matrix(out[1].get_block_(b)).data, expected_v) << context;
              }
            }
          }
        }
      }

      /*=====test info=====
      describe:On the CPU "default" is the QR driver: it returns bit for bit what "qr" returns,
        eigenvectors included, on a matrix large enough for divide and conquer to take its own
        path.
      ====================*/
      TEST(Eigh, CpuDefaultIsQr) {
        for (unsigned int dtype : kDtypes) {
          const Case c = cases_for(dtype)[2];  // dense 33
          const Tensor a = eigh::to_tensor(c.matrix, dtype);
          const std::vector<Tensor> by_default = linalg::Eigh(a);
          const std::vector<Tensor> by_name = linalg::Eigh(a, true, false, "qr");
          ASSERT_EQ(by_default.size(), 2u);
          EXPECT_EQ(eigh::to_values(by_default[0]), eigh::to_values(by_name[0]));
          EXPECT_EQ(eigh::to_matrix(by_default[1]).data, eigh::to_matrix(by_name[1]).data)
            << "dtype " << Type.getname(dtype);
        }
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
