#include <benchmark/benchmark.h>
#include <cytnx.hpp>
using namespace cytnx;

namespace BMTest_Lanczos {
  UniTensor CreateOneSiteEffHam(const int d, const int D, const unsigned int dypte = Type.Double,
                                const int device = Device.cpu);
  UniTensor CreateA(const int d, const int D, const unsigned int dtype = Type.Double,
                    const int device = Device.cpu);
  class OneSiteOp : public LinOp {
   public:
    OneSiteOp(const int d, const int D, const unsigned int dtype = Type.Double,
              const int& device = Device.cpu)
        : LinOp("mv", D * D, dtype, device) {
      EffH = CreateOneSiteEffHam(d, D, dtype, device);
    }
    UniTensor EffH;

    /*
     *         |-|--"vil" "pi" "vir"--|-|                        |-|--"vil" "pi" "vir"--|-|
     *         | |         +          | |             "po"       | |         +          | |
     *         |L|- -------O----------|R|  dot         |       = |L|- -------O----------|R|
     *         | |         +          | |       "vol"--A--"vor"  | |         +          | |
     *         |_|--"vol" "po" "vor"--|_|                        |_|---------A----------|_|
     *
     * Then relabels ["vil", "pi", "vir"] -> ["vol", "po", "vor"]
     *
     * "vil":virtual in bond left
     * "po":physical out bond
     */
    UniTensor matvec(const UniTensor& A) override {
      auto tmp = Contract(EffH, A);
      tmp.permute_({"vil", "pi", "vir"}, 1);
      tmp.relabels_(A.labels());
      return tmp;
    }
  };

  // describe:test not supported UniTensor Type

  /*
   *     -1
   *     |
   *  0--A--2
   */
  UniTensor CreateA(const int d, const int D, const unsigned int dtype, const int device) {
    double low = -1.0, high = 1.0;
    UniTensor A = UniTensor({Bond(D), Bond(d), Bond(D)}, {}, -1, dtype, device)
                    .set_name("A")
                    .relabels_({"vol", "po", "vor"})
                    .set_rowrank_(1);
    if (Type.is_float(A.dtype())) {
      random::uniform_(A, low, high, 0);
    }
    return A;
  }

  /*
   *         |-|--"vil" "pi" "vir"--|-|
   *         | |         +          | |
   *         |L|- -------O----------|R|
   *         | |         +          | |
   *         |_|--"vol" "po" "vor"--|_|
   */
  UniTensor CreateOneSiteEffHam(const int d, const int D, const unsigned int dtype,
                                const int device) {
    double low = -1.0, high = 1.0;
    std::vector<Bond> bonds = {Bond(D), Bond(d), Bond(D), Bond(D), Bond(d), Bond(D)};
    std::vector<std::string> heff_labels = {"vil", "pi", "vir", "vol", "po", "vor"};
    UniTensor HEff = UniTensor(bonds, {}, -1, dtype, device)
                       .set_name("HEff")
                       .relabels_(heff_labels)
                       .set_rowrank(bonds.size() / 2);
    auto HEff_shape = HEff.shape();
    auto in_dim = 1;
    for (int i = 0; i < HEff.rowrank(); ++i) {
      in_dim *= HEff_shape[i];
    }
    auto out_dim = in_dim;
    if (Type.is_float(HEff.dtype())) {
      random::uniform_(HEff, low, high, 0);
    }
    auto HEff_mat = HEff.get_block();
    HEff_mat.reshape_({in_dim, out_dim});
    HEff_mat = HEff_mat + HEff_mat.permute({1, 0});  // symmtrize

    // Let H can be converge in ExpM
    auto eigs = HEff_mat.Eigh();
    auto e = UniTensor(eigs[0], true) * 0.01;
    e.set_labels({"a", "b"});
    auto v = UniTensor(eigs[1]);
    v.set_labels({"i", "a"});
    auto vt = UniTensor(linalg::InvM(v.get_block()));
    vt.set_labels({"b", "j"});
    HEff_mat = Contract(Contract(e, v), vt).get_block();

    // HEff_mat = linalg::Matmul(HEff_mat, HEff_mat.permute({1, 0}).Conj());  // positive definete
    HEff_mat.reshape_(HEff_shape);
    HEff.put_block(HEff_mat);
    return HEff;
  }

  static void BM_Lanczos_Gnd_F64(benchmark::State& state) {
    // prepare data
    int d = 2;
    auto D = state.range(0);
    auto op = OneSiteOp(d, D);
    auto Tin = CreateA(d, D);
    const double crit = 1.0e+8;
    const int maxiter = 2;
    bool is_V = true;
    int k = 1;
    bool is_row = false;
    int max_krydim = 0;
    // start test here
    for (auto _ : state) {
      auto x = linalg::Lanczos(&op, Tin, "Gnd", crit, maxiter, k, is_V, is_row, max_krydim);
    }
  }
  BENCHMARK(BM_Lanczos_Gnd_F64)->Args({10})->Args({30})->Unit(benchmark::kMillisecond);

  static void BM_Lanczos_Exp_F64(benchmark::State& state) {
    // prepare data
    int d = 2;
    auto D = state.range(0);
    auto op = OneSiteOp(d, D);
    auto Tin = CreateA(d, D);
    const double crit = 1.0e+8;
    double tau = 0.1;
    const int maxiter = 2;
    // start test here
    for (auto _ : state) {
      auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit, maxiter);
    }
  }
  BENCHMARK(BM_Lanczos_Exp_F64)->Args({10})->Args({30})->Unit(benchmark::kMillisecond);

}  // namespace BMTest_Lanczos
