#include <cmath>
#include <complex>
#include <limits>
#include <string>

#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "test_tools.h"

namespace cytnx {
  namespace test {

    static UniTensor CreateOneSiteEffHam(const int d, const int D,
                                         const unsigned int dypte = Type.Double,
                                         const int device = Device.cpu);
    static UniTensor CreateA(const int d, const int D, const unsigned int dtype = Type.Double,
                             const int device = Device.cpu);
    static UniTensor GetAns(const UniTensor& HEff, const UniTensor& Tin, const Scalar& tau);
    static Scalar Dot(const UniTensor& A, const UniTensor& B) {
      return Contract(A.Dagger(), B).item();
    }
    class LanczosExpOneSiteOp : public LinOp {
     public:
      LanczosExpOneSiteOp(const int d, const int D, const unsigned int dtype = Type.Double,
                          const int& device = Device.cpu)
          : LinOp("mv", D * d * D, dtype, device) {
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
       * Then relabel ["vil", "pi", "vir"] -> ["vol", "po", "vor"]
       *
       * "vil":virtual in bond left
       * "po":physical out bond
       */
      UniTensor matvec(const UniTensor& A) override {
        auto tmp = Contract(EffH, A);
        tmp.permute_({"vil", "pi", "vir"}, 1);
        tmp.relabel_(A.labels());
        return tmp;
      }
    };

    class OneDimScaleOp : public LinOp {
     public:
      OneDimScaleOp() : LinOp("mv", 1, Type.Double, Device.cpu) {}
      UniTensor matvec(const UniTensor& A) override { return A * 3.0; }
    };

    class SmallResidualOp : public LinOp {
     public:
      explicit SmallResidualOp(const double coupling, const unsigned int dtype = Type.Double)
          : LinOp("mv", 3, dtype, Device.cpu), coupling_(coupling) {}

      UniTensor matvec(const UniTensor& A) override {
        auto out = UniTensor::zeros(A.shape(), A.labels(), A.dtype(), A.device());
        out.set_rowrank_(A.rowrank());
        out.at({0, 0}) = coupling_ * A.at({1, 0});
        out.at({1, 0}) = coupling_ * A.at({0, 0});
        return out;
      }

     private:
      double coupling_;
    };

    class TwoDimMixingOp : public LinOp {
     public:
      TwoDimMixingOp() : LinOp("mv", 2, Type.Double, Device.cpu) {}

      UniTensor matvec(const UniTensor& A) override {
        auto out = UniTensor::zeros(A.shape(), A.labels(), A.dtype(), A.device());
        out.set_rowrank_(A.rowrank());
        out.at({0, 0}) = A.at({1, 0});
        out.at({1, 0}) = A.at({0, 0});
        return out;
      }
    };

    double FloatLanczosExpTolerance() { return 100.0 * std::numeric_limits<float>::epsilon(); }

    static bool IsSinglePrecisionDType(const unsigned int dtype) {
      return dtype == Type.Float || dtype == Type.ComplexFloat;
    }

    static UniTensor SmallResidualInitialState(const unsigned int dtype) {
      UniTensor Tin = UniTensor::zeros({3, 1}, {}, dtype, Device.cpu).set_rowrank_(1);
      Tin.at({0, 0}) = 1.0;
      return Tin;
    }

    static UniTensor SmallResidualExpectedState(const double coupling, const double tau,
                                                const unsigned int dtype) {
      auto ans = UniTensor::zeros({3, 1}, {}, dtype, Device.cpu).set_rowrank_(1);
      ans.at({0, 0}) = std::cosh(coupling * tau);
      ans.at({1, 0}) = std::sinh(coupling * tau);
      return ans;
    }

    static UniTensor SmallResidualExpectedState(const double coupling, const cytnx_complex128 tau,
                                                const unsigned int dtype) {
      auto ans = UniTensor::zeros({3, 1}, {}, dtype, Device.cpu).set_rowrank_(1);
      ans.at({0, 0}) = std::cosh(coupling * tau);
      ans.at({1, 0}) = std::sinh(coupling * tau);
      return ans;
    }

    static UniTensor TwoDimInitialState() {
      auto Tin = UniTensor::zeros({2, 1}, {}, Type.Double, Device.cpu).set_rowrank_(1);
      Tin.at({0, 0}) = 1.0;
      return Tin;
    }

    static UniTensor TwoDimExpectedState(const double tau) {
      auto ans = UniTensor::zeros({2, 1}, {}, Type.Double, Device.cpu).set_rowrank_(1);
      ans.at({0, 0}) = std::cosh(tau);
      ans.at({1, 0}) = std::sinh(tau);
      return ans;
    }

    // describe:Real type test
    TEST(LanczosExpUt, RealTypeTest) {
      int d = 2, D = 5;
      auto op = LanczosExpOneSiteOp(d, D);
      auto Tin = CreateA(d, D);
      const double crit = 1.0e-10;
      double tau = 0.1;
      auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit);
      auto ans = GetAns(op.EffH, Tin, tau);
      auto err = static_cast<double>((x - ans).Norm().item().real());
      EXPECT_LE(err, crit);
    }

    // describe:Complex type test
    TEST(LanczosExpUt, ComplexTypeTest) {
      int d = 2, D = 5;
      auto op = LanczosExpOneSiteOp(d, D, Type.ComplexDouble);
      auto Tin = CreateA(d, D, Type.ComplexDouble);
      const double crit = 1.0e-9;
      std::complex<double> tau = std::complex<double>(0, 1) * 0.1;
      auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit);
      auto ans = GetAns(op.EffH, Tin, tau);
      auto err = static_cast<double>((x - ans).Norm().item().real());
      EXPECT_LE(err, crit);
    }

    // describe:Test non-Hermitian Op but the code will not crash
    TEST(LanczosExpUt, NonHermit) {
      int d = 2, D = 5;
      double low = -1.0, high = 1.0;
      auto op = LanczosExpOneSiteOp(d, D);
      op.EffH.uniform_(low, high, 0);
      auto Tin = CreateA(d, D);
      const double crit = 1.0e-3;
      double tau = 0.1;
      auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit);
    }

    // describe:input |v| != 1
    TEST(LanczosExpUt, NormVNot1) {
      int d = 2, D = 5;
      auto op = LanczosExpOneSiteOp(d, D);
      auto Tin = CreateA(d, D) * 1.1;
      const double crit = 1.0e-7;
      double tau = 0.1;
      auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit);
      auto ans = GetAns(op.EffH, Tin, tau);
      auto err = static_cast<double>((x - ans).Norm().item().real());
      EXPECT_LE(err, crit);
    }

    TEST(LanczosExpUt, OneDimensionalKrylovSpace) {
      OneDimScaleOp op;
      UniTensor Tin = UniTensor::zeros({1, 1}, {}, Type.Double, Device.cpu).set_rowrank_(1);
      Tin.at({0, 0}) = 2.0;
      const double crit = 1.0e-12;
      const double tau = 0.2;

      auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit);
      auto ans = Tin * std::exp(3.0 * tau);
      auto err = static_cast<double>((x - ans).Norm().item().real());

      EXPECT_LE(err, crit);
    }

    TEST(LanczosExpUt, FullKrylovSpaceDoesNotWarnAtDimensionLimit) {
      TwoDimMixingOp op;
      auto Tin = TwoDimInitialState();
      const double crit = 1.0e-12;
      const double tau = 1.0;
      const unsigned int maxiter = 2;

      testing::internal::CaptureStderr();
      auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit, maxiter);
      const std::string stderr_output = testing::internal::GetCapturedStderr();
      auto ans = TwoDimExpectedState(tau);
      auto err = static_cast<double>((x - ans).Norm().item().real());

      EXPECT_EQ(stderr_output.find("[WARNING][Lanczos_Exp]"), std::string::npos) << stderr_output;
      EXPECT_LE(err, crit);
    }

    TEST(LanczosExpUt, SmallResidualIsNotBreakdown) {
      const double coupling = 5.0e-7;
      SmallResidualOp op(coupling);
      auto Tin = SmallResidualInitialState(Type.Double);
      const double crit = 1.0e-10;
      const double tau = 1.0;
      const unsigned int maxiter = 3;

      auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit, maxiter);
      auto ans = SmallResidualExpectedState(coupling, tau, Type.Double);
      auto err = static_cast<double>((x - ans).Norm().item().real());

      EXPECT_LE(err, crit);
    }

    TEST(LanczosExpUt, FloatSmallResidualBelowRoundoffFloor) {
      const double coupling = 5.0e-6;
      SmallResidualOp op(coupling, Type.Float);
      auto Tin = SmallResidualInitialState(Type.Float);
      const double tau = 1.0;
      const unsigned int maxiter = 3;

      auto x = linalg::Lanczos_Exp(&op, Tin, tau, 1.0e-10, maxiter);
      auto ans = SmallResidualExpectedState(coupling, tau, x.dtype());
      auto err = static_cast<double>((x - ans).Norm().item().real());

      // ExpM currently returns a complex matrix even for this real case. Either
      // real or complex output is acceptable here, but it must remain single precision.
      EXPECT_TRUE(IsSinglePrecisionDType(x.dtype()));
      EXPECT_LE(err, FloatLanczosExpTolerance());
    }

    TEST(LanczosExpUt, FloatResidualAboveRoundoffFloorIsNotBreakdown) {
      const double coupling = 5.0e-5;
      SmallResidualOp op(coupling, Type.Float);
      auto Tin = SmallResidualInitialState(Type.Float);
      const double tau = 1.0;
      const unsigned int maxiter = 3;

      auto x = linalg::Lanczos_Exp(&op, Tin, tau, 1.0e-10, maxiter);
      auto ans = SmallResidualExpectedState(coupling, tau, x.dtype());
      auto err = static_cast<double>((x - ans).Norm().item().real());

      // ExpM currently returns a complex matrix even for this real case. Either
      // real or complex output is acceptable here, but it must remain single precision.
      EXPECT_TRUE(IsSinglePrecisionDType(x.dtype()));
      EXPECT_LE(err, FloatLanczosExpTolerance());
    }

    TEST(LanczosExpUt, ComplexFloatResidualAboveRoundoffFloorIsNotBreakdown) {
      const double coupling = 5.0e-5;
      SmallResidualOp op(coupling, Type.ComplexFloat);
      auto Tin = SmallResidualInitialState(Type.ComplexFloat);
      const double tau = 1.0;
      const unsigned int maxiter = 3;

      auto x = linalg::Lanczos_Exp(&op, Tin, tau, 1.0e-10, maxiter);
      auto ans = SmallResidualExpectedState(coupling, tau, Type.ComplexFloat);
      auto err = static_cast<double>((x - ans).Norm().item().real());

      EXPECT_EQ(x.dtype(), Type.ComplexFloat);
      EXPECT_LE(err, FloatLanczosExpTolerance());
    }

    TEST(LanczosExpUt, FloatComplexTauReturnsComplexFloat) {
      const double coupling = 5.0e-5;
      SmallResidualOp op(coupling, Type.Float);
      auto Tin = SmallResidualInitialState(Type.Float);
      const cytnx_complex128 tau(0.0, 1.0);
      const unsigned int maxiter = 3;

      auto x = linalg::Lanczos_Exp(&op, Tin, tau, 1.0e-10, maxiter);
      auto ans = SmallResidualExpectedState(coupling, tau, Type.ComplexFloat);
      auto err = static_cast<double>((x - ans).Norm().item().real());

      EXPECT_EQ(x.dtype(), Type.ComplexFloat);
      EXPECT_LE(err, FloatLanczosExpTolerance());
    }

    // describe:test incorrect data type
    TEST(LanczosExpUt, IncorrectDType) {
      int d = 2, D = 10;
      auto op = LanczosExpOneSiteOp(d, D, Type.Int64);
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
    static UniTensor CreateA(const int d, const int D, const unsigned int dtype, const int device) {
      double low = -1.0, high = 1.0;
      UniTensor A = UniTensor({Bond(D), Bond(d), Bond(D)}, {}, -1, dtype, device)
                      .set_name("A")
                      .relabel_({"vol", "po", "vor"})
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
    static UniTensor CreateOneSiteEffHam(const int d, const int D, const unsigned int dtype,
                                         const int device) {
      double low = -1.0, high = 1.0;
      std::vector<Bond> bonds = {Bond(D), Bond(d), Bond(D), Bond(D), Bond(d), Bond(D)};
      std::vector<std::string> heff_labels = {"vil", "pi", "vir", "vol", "po", "vor"};
      UniTensor HEff = UniTensor(bonds, {}, -1, dtype, device)
                         .set_name("HEff")
                         .relabel_(heff_labels)
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
      e.relabel_({"a", "b"});
      auto v = UniTensor(eigs[1]);
      v.relabel_({"i", "a"});
      auto vt = UniTensor(linalg::InvM(v.get_block()));
      vt.relabel_({"b", "j"});
      HEff_mat = Contract(Contract(e, v), vt).get_block();

      // HEff_mat = linalg::Matmul(HEff_mat, HEff_mat.permute({1, 0}).Conj());  // positive
      // definete
      HEff_mat.reshape_(HEff_shape);
      HEff.put_block(HEff_mat);
      return HEff;
    }

    static UniTensor GetAns(const UniTensor& HEff, const UniTensor& Tin, const Scalar& tau) {
      auto expH = HEff.clone();
      auto HEff_shape = HEff.shape();
      auto in_dim = 1;
      for (int i = 0; i < HEff.rowrank(); ++i) {
        in_dim *= HEff_shape[i];
      }
      auto out_dim = in_dim;
      // we use ExpM since tau*H will not be Hermitian if tau is complex number even H is
      // Hermitian
      expH.put_block(
        linalg::ExpM((tau * expH.get_block()).reshape(in_dim, out_dim)).reshape(HEff_shape));
      auto ans = Contract(expH, Tin);
      ans.permute_({"vil", "pi", "vir"}, 1);
      ans.relabel_(Tin.labels());
      ans = Contract(expH, Tin);
      return ans;
    }

  }  // namespace test
}  // namespace cytnx
