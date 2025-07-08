#include <benchmark/benchmark.h>
#include <cytnx.hpp>
using namespace cytnx;

namespace BMTest_Arnoldi {
  class TMOp : public LinOp {
   public:
    UniTensor A, B;
    UniTensor T_init;
    TMOp(const int& d, const int& D, const cytnx_uint64& nx,
         const unsigned int& dtype = Type.Double, const int& device = Device.cpu);
    UniTensor matvec(const UniTensor& l) override {
      auto tmp = Contracts({A, l, B}, "", true);
      tmp.relabels_(l.labels()).set_rowrank(l.rowrank());
      return tmp;
    }
  };

  TMOp::TMOp(const int& d, const int& D, const cytnx_uint64& in_nx, const unsigned int& in_dtype,
             const int& in_device)
      : LinOp("mv", in_nx, in_dtype, in_device) {
    std::vector<Bond> bonds = {Bond(D), Bond(d), Bond(D)};
    A = UniTensor(bonds, {}, -1, in_dtype, in_device)
          .set_name("A")
          .relabels_({"al", "phys", "ar"})
          .set_rowrank(2);
    B = UniTensor(bonds, {}, -1, in_dtype, in_device)
          .set_name("B")
          .relabels_({"bl", "phys", "br"})
          .set_rowrank(2);
    T_init = UniTensor({Bond(D), Bond(D)}, {}, -1, in_dtype, in_device)
               .set_name("l")
               .relabels_({"al", "bl"})
               .set_rowrank(1);
    if (Type.is_float(this->dtype())) {
      double low = -1.0, high = 1.0;
      int seed = 0;
      A.uniform_(low, high, seed);
      B.uniform_(low, high, seed);
      T_init.uniform_(low, high, seed);
    }
  }

  static void BM_Arnoldi_Ut_C128(benchmark::State& state) {
    // prepare data
    std::string which = "LM";
    int d = 2;
    auto D = state.range(0);
    int k = state.range(1);
    int dim = D * D;
    auto op = TMOp(d, D, dim, Type.ComplexDouble);
    const double crit = 0;
    const int maxiter = 1000;
    // start test here
    for (auto _ : state) {
      std::vector<UniTensor> arnoldi_eigs =
        linalg::Arnoldi(&op, op.T_init, which, maxiter, crit, k);
    }
  }
  BENCHMARK(BM_Arnoldi_Ut_C128)
    ->Args({10, 1})
    ->Args({20, 1})
    ->Args({40, 1})
    ->Args({10, 4})
    ->Args({20, 4})
    ->Args({40, 4})
    ->Unit(benchmark::kMillisecond);

  static void BM_Arnoldi_Ut_F64(benchmark::State& state) {
    // prepare data
    std::string which = "LM";
    int d = 2;
    auto D = state.range(0);
    int k = state.range(1);
    int dim = D * D;
    auto op = TMOp(d, D, dim, Type.Double);
    const double crit = 0;
    const int maxiter = 1000;
    // start test here
    for (auto _ : state) {
      std::vector<UniTensor> arnoldi_eigs =
        linalg::Arnoldi(&op, op.T_init, which, maxiter, crit, k);
    }
  }
  BENCHMARK(BM_Arnoldi_Ut_F64)
    ->Args({10, 1})
    ->Args({20, 1})
    ->Args({40, 1})
    ->Args({10, 4})
    ->Args({20, 4})
    ->Args({40, 4})
    ->Unit(benchmark::kMillisecond);

}  // namespace BMTest_Arnoldi
