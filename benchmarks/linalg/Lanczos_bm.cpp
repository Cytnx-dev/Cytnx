#include <benchmark/benchmark.h>
#include <cytnx.hpp>
using namespace cytnx;

namespace BMTest_Lanczos {
  class OneSiteOp : public LinOp {
   public:
    OneSiteOp(const int d = 2, const int D = 5, const int dw = 3,
              const unsigned int dtype = Type.Double, const int& device = Device.cpu)
        : LinOp("mv", D * D * d, dtype, device) {
      if (!Type.is_float(dtype)) return;
      std::vector<UniTensor> LRO = CreateLRO(D, d, dw);
      L_ = LRO[0];
      R_ = LRO[1];
      O_ = LRO[2];
      net_.FromString({"psi:'vil', 'pi', 'vir'", "L:'vil', 'Lm', 'vol'", "O:'Lm', 'pi', 'Rm', 'po'",
                       "R:'vir', 'Rm', 'vor'", "TOUT:'vol'; 'po', 'vor'"});
      net_.PutUniTensors({"L", "O", "R"}, {L_, O_, R_});
      UT_init = Create_UTinit(D, d);
    }
    UniTensor UT_init;

    /*
     *         |-|--"vil" "pi" "vir"--|-|
     *         | |         +          | |      "vil"--psi--"vir"
     *         |L|--"Lm"---O----"Rm"--|R|  dot         |
     *         | |         +          | |             "pi"
     *         |_|--"vol" "po" "vor"--|_|
     *
     * Then relabels ["vil", "pi", "vir"] -> ["vol", "po", "vor"]
     *
     * "vil":virtual in bond left
     * "po":physical out bond
     */

    UniTensor matvec(const UniTensor& psi) override {
      net_.PutUniTensor("psi", psi);
      return net_.Launch().relabel_(psi.labels());
    }

   private:
    UniTensor L_, R_, O_;
    Network net_;
    std::vector<UniTensor> CreateLRO(const int D, const int d, const int dw) {
      double low = -1.0, high = 1.0;
      int seed = 0;
      UniTensor L =
        UniTensor::uniform({D, dw, D}, low, high, {"vil", "Lm", "vol"}, seed, dtype(), device());
      seed = 1;
      UniTensor R =
        UniTensor::uniform({D, dw, D}, low, high, {"vir", "Rm", "vor"}, seed, dtype(), device());
      seed = 1;
      UniTensor O = UniTensor::uniform({dw, d, dw, d}, low, high, {"Lm", "pi", "Rm", "po"}, seed,
                                       dtype(), device());
      L = L + L.permute({"vol", "Lm", "vil"}).Conj().contiguous();
      R = R + R.permute({"vor", "Rm", "vir"}).Conj().contiguous();
      O = O + O.permute({"Lm", "po", "Rm", "pi"}).Conj().contiguous();
      return {L, R, O};
    }

    UniTensor Create_UTinit(const int D, const int d) {
      double low = -1.0, high = 1.0;
      int seed = 0;
      UniTensor psi =
        UniTensor::uniform({D, d, D}, low, high, {"vil", "pi", "vir"}, seed, dtype(), device());
      return psi;
    }
  };

  static void BM_Lanczos_Gnd_F64(benchmark::State& state) {
    // prepare data
    int d = 2;
    int dw = 5;
    auto D = state.range(0);
    auto op = OneSiteOp(d, D, dw, Type.Double);
    const double crit = 1.0e-12;
    const int maxiter = 1000;
    bool is_V = true;
    int k = 1;
    bool is_row = false;
    int max_krydim = 0;
    // start test here
    for (auto _ : state) {
      auto x = linalg::Lanczos(&op, op.UT_init, "Gnd", crit, maxiter, k, is_V, is_row, max_krydim);
    }
  }
  BENCHMARK(BM_Lanczos_Gnd_F64)
    ->Args({10})
    ->Args({20})
    ->Args({50})
    ->Args({100})
    ->Unit(benchmark::kMillisecond);

  static void BM_Lanczos_Ut_F64(benchmark::State& state) {
    // prepare data
    int d = 2;
    int dw = 5;
    auto D = state.range(0);
    auto op = OneSiteOp(d, D, dw, Type.Double);
    const double crit = 1.0e-12;
    const int maxiter = 1000;
    bool is_V = true;
    int k = 1;
    int max_krydim = 0;
    // start test here
    for (auto _ : state) {
      auto x = linalg::Lanczos(&op, op.UT_init, "SA", maxiter, crit, k, is_V);
    }
  }
  BENCHMARK(BM_Lanczos_Ut_F64)
    ->Args({10})
    ->Args({20})
    ->Args({50})
    ->Args({100})
    ->Unit(benchmark::kMillisecond);

  static void BM_Lanczos_Exp_F64(benchmark::State& state) {
    // prepare data
    int d = 2;
    int dw = 3;
    auto D = state.range(0);
    auto op = OneSiteOp(d, D, dw);
    const double crit = 1.0e-8;
    double tau = 0.1;
    const int maxiter = 1000;
    // start test here
    for (auto _ : state) {
      auto x = linalg::Lanczos_Exp(&op, op.UT_init, tau, crit, maxiter);
    }
  }
  BENCHMARK(BM_Lanczos_Exp_F64)->Args({10})->Args({30})->Unit(benchmark::kMillisecond);

}  // namespace BMTest_Lanczos
