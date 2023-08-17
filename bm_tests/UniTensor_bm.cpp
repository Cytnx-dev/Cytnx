#include <benchmark/benchmark.h>
#include <cytnx.hpp>

// Cytnx test
namespace BMTest_UniTensor {
  std::vector<cytnx::UniTensor> ConstructDenseUT(const long unsigned int D,
                                                 const unsigned int dtype, const int device) {
    // H|psi> in dmrg progress:
    /*
       +--- psi ----+
           |   |    |   |
           L---M1--M2---R
           |   |    |   |
        */
    std::vector<long unsigned int> shape;
    std::vector<std::string> labels;
    double low, high;
    int rowrank;
    bool is_diag;
    unsigned int seed;
    std::string name;

    auto T =
      cytnx::random::uniform(shape = {4, D, D}, low = -1.0, high = 1.0, device, seed = 0, dtype);

    cytnx::UniTensor L =
      cytnx::UniTensor(T, is_diag = false, rowrank = -1, labels = {"-5", "-1", "0"}, name = "L");

    cytnx::UniTensor R =
      cytnx::UniTensor(T, is_diag = false, rowrank = -1, labels = {"-7", "-4", "3"}, name = "R");

    T =
      cytnx::random::uniform(shape = {4, 4, 2, 2}, low = -1.0, high = 1.0, device, seed = 0, dtype);

    cytnx::UniTensor M1 =
      cytnx::UniTensor(T, is_diag = false, rowrank = -1, labels = {"-5", "-6", "-2", "1"}, "M1");

    cytnx::UniTensor M2 =
      cytnx::UniTensor(T, is_diag = false, rowrank = -1, labels = {"-6", "-7", "-3", "2"}, "M2");

    T =
      cytnx::random::uniform(shape = {D, 2, 2, D}, low = -1.0, high = 1.0, device, seed = 0, dtype);

    cytnx::UniTensor psi =
      cytnx::UniTensor(T, is_diag = false, rowrank = -1, labels = {"-1", "-2", "-3", "-4"}, "psi");
    return {L, R, M1, M2, psi};
  }

  static void Cytnx_Hpsi_dense_F64_cpu(benchmark::State& state) {
    long unsigned int D = state.range(0);
    auto dtype = cytnx::Type.Double;
    auto device = cytnx::Device.cpu;
    std::vector<cytnx::UniTensor> unitens = ConstructDenseUT(D, dtype, device);
    auto &L = unitens[0], R = unitens[1], M1 = unitens[2], M2 = unitens[3], psi = unitens[4];
    for (auto _ : state) {
      auto out = cytnx::Contracts({L, R, M1, M2, psi}, "(L,(M1,(M2,(psi,R))))", false);
    }
  }
  BENCHMARK(Cytnx_Hpsi_dense_F64_cpu)->Args({1})->Args({10})->Args({100})->Args({1000});

  static void Cytnx_Hpsi_dense_C128_cpu(benchmark::State& state) {
    long unsigned int D = state.range(0);
    auto dtype = cytnx::Type.ComplexDouble;
    auto device = cytnx::Device.cpu;
    std::vector<cytnx::UniTensor> unitens = ConstructDenseUT(D, dtype, device);
    auto &L = unitens[0], R = unitens[1], M1 = unitens[2], M2 = unitens[3], psi = unitens[4];
    for (auto _ : state) {
      auto out = cytnx::Contracts({L, R, M1, M2, psi}, "(L,(M1,(M2,(psi,R))))", false);
    }
  }
  BENCHMARK(Cytnx_Hpsi_dense_C128_cpu)->Args({1})->Args({10})->Args({100})->Args({1000});

#ifdef UNI_GPU
  static void Cytnx_Hpsi_dense_F64_gpu(benchmark::State& state) {
    long unsigned int D = state.range(0);
    auto dtype = cytnx::Type.Double;
    auto device = cytnx::Device.cuda;
    std::vector<cytnx::UniTensor> unitens = ConstructDenseUT(D, dtype, device);
    auto &L = unitens[0], R = unitens[1], M1 = unitens[2], M2 = unitens[3], psi = unitens[4];
    for (auto _ : state) {
      auto out = cytnx::Contracts({L, R, M1, M2, psi}, "(L,(M1,(M2,(psi,R))))", false);
    }
  }
  BENCHMARK(Cytnx_Hpsi_dense_F64_gpu)->Args({1})->Args({10})->Args({100})->Args({1000});

  static void Cytnx_Hpsi_dense_C128_gpu(benchmark::State& state) {
    long unsigned int D = state.range(0);
    auto dtype = cytnx::Type.ComplexDouble;
    auto device = cytnx::Device.cuda;
    std::vector<cytnx::UniTensor> unitens = ConstructDenseUT(D, dtype, device);
    auto &L = unitens[0], R = unitens[1], M1 = unitens[2], M2 = unitens[3], psi = unitens[4];
    for (auto _ : state) {
      auto out = cytnx::Contracts({L, R, M1, M2, psi}, "(L,(M1,(M2,(psi,R))))", false);
    }
  }
  BENCHMARK(Cytnx_Hpsi_dense_C128_gpu)->Args({1})->Args({10})->Args({100})->Args({1000});
#endif

  // test with several arguments, ex, bond dimension
  static void BM_ones(benchmark::State& state) {
    int num_1 = state.range(0);
    int num_2 = state.range(1);
    for (auto _ : state) {
      auto A = cytnx::UniTensor(cytnx::ones({num_1, num_2, 3}));
    }
  }

  BENCHMARK(BM_ones)->Args({5, 3})->Args({10, 9});

}  // namespace BMTest_UniTensor
