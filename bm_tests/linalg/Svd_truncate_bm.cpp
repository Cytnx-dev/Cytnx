#include <benchmark/benchmark.h>
#include <cytnx.hpp>
using namespace cytnx;

namespace BMTest_Svd_truncate {

  // DenseUniTensor
  UniTensor ConstructDenseUT(const int D, const int dtype, const unsigned int device = Device.cpu) {
    auto bd_vi = Bond(D, BD_IN);
    auto bd_pi = Bond(2, BD_IN);
    auto bd_po = Bond(2, BD_OUT);
    auto bd_vo = Bond(D, BD_OUT);
    std::vector<Bond> bonds = {bd_vi, bd_pi, bd_po, bd_vo};
    cytnx_int64 row_rank = 2;
    std::vector<std::string> labels = {};
    bool is_diag = false;
    auto UT = UniTensor(bonds, labels, row_rank, dtype, device, is_diag);
    double low = -1000, high = 1000;
    int rnd_seed = 0;
    random::Make_uniform(UT, low = -1000, high = 1000, rnd_seed);
    return UT;
  }

  static void BM_DenseUT_Svd_truncate_F64(benchmark::State& state) {
    // prepare data
    auto D = state.range(0);
    UniTensor UT = ConstructDenseUT(D, Type.Double);

    // start test here
    for (auto _ : state) {
      std::vector<UniTensor> svds = linalg::Svd_truncate(UT, D);
    }
  }
  BENCHMARK(BM_DenseUT_Svd_truncate_F64)->Args({1})->Args({5})->Args({50})->Args({500});

  static void BM_DenseUT_Svd_truncate_C128(benchmark::State& state) {
    // prepare data
    auto D = state.range(0);
    UniTensor UT = ConstructDenseUT(D, Type.ComplexDouble);

    // start test here
    for (auto _ : state) {
      std::vector<UniTensor> svds = linalg::Svd_truncate(UT, D);
    }
  }
  BENCHMARK(BM_DenseUT_Svd_truncate_C128)->Args({1})->Args({5})->Args({50})->Args({500});

#ifdef UNI_GPU
  static void BM_gpu_DenseUT_Svd_truncate_F64(benchmark::State& state) {
    // prepare data
    auto D = state.range(0);
    UniTensor UT = ConstructDenseUT(D, Type.Double, Device.cuda);

    // start test here
    for (auto _ : state) {
      std::vector<UniTensor> svds = linalg::Svd_truncate(UT, D);
    }
  }
  BENCHMARK(BM_gpu_DenseUT_Svd_truncate_F64)->Args({1})->Args({5})->Args({50});

  static void BM_gpu_DenseUT_Svd_truncate_C128(benchmark::State& state) {
    // prepare data
    auto D = state.range(0);
    UniTensor UT = ConstructDenseUT(D, Type.ComplexDouble, Device.cuda);

    // start test here
    for (auto _ : state) {
      std::vector<UniTensor> svds = linalg::Svd_truncate(UT, D);
    }
  }
  BENCHMARK(BM_gpu_DenseUT_Svd_truncate_C128)->Args({1})->Args({5})->Args({50});
#endif

  // Block UniTensor
  UniTensor ConstructBkUT(const int deg, const int dtype, const int symType, const int n = 0) {
    // construct bonds
    std::vector<std::vector<cytnx_int64>> qnums1 = {{0}, {1}, {0}, {1}, {2}};
    std::vector<cytnx_uint64> degs = std::vector<cytnx_uint64>(qnums1.size(), deg);
    auto syms = std::vector<Symmetry>(qnums1[0].size(), Symmetry(symType, n));
    auto bond_ket = Bond(BD_KET, qnums1, degs, syms);
    std::vector<std::vector<cytnx_int64>> qnums2 = {{-1}, {-1}, {0}, {2}, {1}};
    syms = std::vector<Symmetry>(qnums2[0].size(), Symmetry(symType, n));
    auto bond_bra = Bond(BD_BRA, qnums2, degs, syms);
    std::vector<Bond> bonds = {bond_ket, bond_bra};
    cytnx_int64 row_rank = -1;
    std::vector<std::string> labels = {};
    bool is_diag;
    auto UT = UniTensor(bonds, labels, row_rank, dtype, Device.cpu, is_diag = false);
    double l_bd, h_bd;
    int rnd_seed;
    random::Make_uniform(UT, l_bd = -1000, h_bd = 1000, rnd_seed);
    return UT;
  }

  static void BM_bkUT_U1_Svd_truncate_F64(benchmark::State& state) {
    // prepare data
    auto D = state.range(0);
    UniTensor bkUT = ConstructBkUT(D, Type.Double, SymType.U);

    // start test here
    for (auto _ : state) {
      std::vector<UniTensor> svds = linalg::Svd_truncate(bkUT, D);
    }
  }
  BENCHMARK(BM_bkUT_U1_Svd_truncate_F64)->Args({2})->Args({20})->Args({200});

  static void BM_bkUT_U1_Svd_truncate_C128(benchmark::State& state) {
    // prepare data
    auto D = state.range(0);
    UniTensor bkUT = ConstructBkUT(D, Type.ComplexDouble, SymType.U);

    // start test here
    for (auto _ : state) {
      std::vector<UniTensor> svds = linalg::Svd_truncate(bkUT, D);
    }
  }
  BENCHMARK(BM_bkUT_U1_Svd_truncate_C128)->Args({2})->Args({20})->Args({200});

}  // namespace BMTest_Svd_truncate
