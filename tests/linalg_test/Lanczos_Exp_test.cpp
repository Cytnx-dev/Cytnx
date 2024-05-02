#include "cytnx.hpp"
#include <gtest/gtest.h>
#include "../test_tools.h"

using namespace cytnx;
using namespace testing;
using namespace TestTools;

namespace Lanczos_Exp_Ut_Test {

  UniTensor CreateOneSiteEffHam(const int d, const int D, const unsigned int dypte = Type.Double,
                                const int device = Device.cpu);
  UniTensor CreateA(const int d, const int D, const unsigned int dtype = Type.Double,
                    const int device = Device.cpu);
  UniTensor GetAns(const UniTensor& HEff, const UniTensor& Tin, const Scalar& tau);
  Scalar Dot(const UniTensor& A, const UniTensor& B) { return Contract(A.Dagger(), B).item(); }
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

  // describe:Real type test
  TEST(Lanczos_Exp_Ut, RealTypeTest) {
    int d = 2, D = 5;
    auto op = OneSiteOp(d, D);
    auto Tin = CreateA(d, D);
    const double crit = 1.0e-10;
    double tau = 0.1;
    auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit);
    auto ans = GetAns(op.EffH, Tin, tau);
    auto err = static_cast<double>((x - ans).Norm().item().real());
    EXPECT_TRUE(err <= crit);
  }

  // describe:Complex type test
  TEST(Lanczos_Exp_Ut, ComplexTypeTest) {
    int d = 2, D = 5;
    auto op = OneSiteOp(d, D, Type.ComplexDouble);
    auto Tin = CreateA(d, D, Type.ComplexDouble);
    const double crit = 1.0e-9;
    std::complex<double> tau = std::complex<double>(0, 1) * 0.1;
    auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit);
    auto ans = GetAns(op.EffH, Tin, tau);
    auto err = static_cast<double>((x - ans).Norm().item().real());
    EXPECT_TRUE(err <= crit);
  }

  // describe:Test non-Hermitian Op but the code will not crash
  TEST(Lanczos_Exp_Ut, NonHermit) {
    int d = 2, D = 5;
    double low = -1.0, high = 1.0;
    auto op = OneSiteOp(d, D);
    op.EffH.uniform_(low, high, 0);
    auto Tin = CreateA(d, D);
    const double crit = 1.0e-3;
    double tau = 0.1;
    auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit);
  }

  // describe:input |v| != 1
  TEST(Lanczos_Exp_Ut, normVNot1) {
    int d = 2, D = 5;
    auto op = OneSiteOp(d, D);
    auto Tin = CreateA(d, D) * 1.1;
    const double crit = 1.0e-7;
    double tau = 0.1;
    auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit);
    auto ans = GetAns(op.EffH, Tin, tau);
    auto err = static_cast<double>((x - ans).Norm().item().real());
    EXPECT_TRUE(err <= crit);
  }

  // describe:test incorrect data type
  TEST(Lanczos_Exp_Ut, IncorrectDType) {
    int d = 2, D = 10;
    auto op = OneSiteOp(d, D, Type.Int64);
    auto Tin = CreateA(d, D, Type.Int64);
    const double crit = 1.0e-3;
    double tau = 0.1;
    EXPECT_THROW({ linalg::Lanczos_Exp(&op, Tin, crit, tau); }, std::logic_error);
  }

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
    // A = A / std::sqrt(double((Dot(A, A).real())));
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

  UniTensor GetAns(const UniTensor& HEff, const UniTensor& Tin, const Scalar& tau) {
    auto expH = HEff.clone();
    auto HEff_shape = HEff.shape();
    auto in_dim = 1;
    for (int i = 0; i < HEff.rowrank(); ++i) {
      in_dim *= HEff_shape[i];
    }
    auto out_dim = in_dim;
    // we use ExpM since tau*H will not be Hermitian if tau is complex number even H is Hermitian
    expH.put_block(
      linalg::ExpM((tau * expH.get_block()).reshape(in_dim, out_dim)).reshape(HEff_shape));
    auto ans = Contract(expH, Tin);
    ans.permute_({"vil", "pi", "vir"}, 1);
    ans.relabels_(Tin.labels());
    ans = Contract(expH, Tin);
    return ans;
  }

}  // namespace Lanczos_Exp_Ut_Test
