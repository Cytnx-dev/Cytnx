#include <gtest/gtest.h>

#include "cytnx.hpp"
#include "gpu_test_tools.h"
#include "linalg_test.h"

namespace cytnx {
  namespace gpu_test {
    namespace {
      using gpu_test::ferm_dense_lowest;
      using gpu_test::FermiAdaOp;
      using gpu_test::make_ferm_ada_ket;

      class MyOp : public LinOp {
       public:
        MyOp() : LinOp("mv", 27) {}

        UniTensor matvec(const UniTensor& v) override {
          Tensor tA = arange(0, 27 * 27, 1, Type.Double, Device.cuda).reshape(27, 27);
          UniTensor A = UniTensor(tA).to(Device.cuda);
          A = A + A.Transpose();
          return UniTensor(linalg::Dot(A.get_block_(), v.get_block_())).to(Device.cuda);
          // return UniTensor(linalg::Tensordot(A.get_block_(), v.get_block_(), {1}, {0}))
          //   .to(Device.cuda);
        }
      };

      class MyOp2 : public LinOp {
       public:
        UniTensor H;
        MyOp2(int dim) : LinOp("mv", dim) {
          Tensor A = Tensor::Load(CYTNX_TEST_DATA_DIR "/linalg/Lanczos_Gnd/lan_block_A.cytn");
          Tensor B = Tensor::Load(CYTNX_TEST_DATA_DIR "/linalg/Lanczos_Gnd/lan_block_B.cytn");
          Tensor C = Tensor::Load(CYTNX_TEST_DATA_DIR "/linalg/Lanczos_Gnd/lan_block_C.cytn");
          Bond lan_I = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
          Bond lan_J = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
          H = UniTensor({lan_I, lan_J});
          H.put_block(A, 0);
          H.put_block(B, 1);
          H.put_block(C, 2);
          H.relabel_({"a", "b"});
          H.to_(Device.cuda);
          // H.print_diagram();
          // H.print_blocks();
        }
        UniTensor matvec(const UniTensor& psi) override {
          auto out = H.contract(psi);
          out.relabel_({"b", "c"});
          return out;
        }
      };

      TEST(LanczosGnd, GpuLanczosGndTest) {
        // CompareWithScipy
        // cytnx_double evans = -0.6524758424985271;
        cytnx_double evans = -1628.9964650426593;

        MyOp H = MyOp();
        Tensor tv = arange(0, 27, 1, Type.Double, Device.cuda);
        UniTensor v = UniTensor(tv).to(Device.cuda);
        std::vector<UniTensor> eigs =
          linalg::Lanczos(&H, v, "Gnd", 9.999999999999999988e-15, 10000, 1, false, true, 0, false);
        cytnx_double ev = (cytnx_double)eigs[0].get_block_().item().real();
        std::cout << ev << ' ' << evans << std::endl;
        EXPECT_TRUE(std::fabs(ev - evans) < 1e-5);
        // EXPECT_DOUBLE_EQ(ev, evans);
      }

      // GPU fermionic Krylov: O = A^dag A (sign-flip-active 4-leg A); eigenpairs computed on the
      // GPU and checked against an independent CPU dense diagonalization.
      TEST(LanczosGnd, GpuFermionicLanczos) {  // naive Lanczos 'Gnd', ground state
        const double tol = 1e-7;
        UniTensor A = make_ferm_A();
        UniTensor v0 = make_ferm_ada_ket(A);
        auto low = ferm_dense_lowest(A, 1);
        FermiAdaOp op(A.to(Device.cuda), ferm_ket_nx(v0));
        auto eigs = to_cpu(linalg::Lanczos(&op, v0.to(Device.cuda), "Gnd", 1e-12, 1000, 1, true));
        expect_lowest_states(A, eigs, low, tol);
      }

      TEST(LanczosGnd, GpuBkLanczosGndTest) {
        // CompareWithScipy
        cytnx_double evans = -2.31950925;

        Bond lan_I_v = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
        Bond lan_J_v = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {1, 1, 1});
        UniTensor lan_guess = UniTensor({lan_I_v, lan_J_v});

        lan_guess.put_block(random::normal(9, 1, 1).reshape({9, 1}), 0);
        lan_guess.put_block(random::normal(9, 1, 1).reshape({9, 1}), 1);
        lan_guess.put_block(random::normal(9, 1, 1).reshape({9, 1}), 2);
        lan_guess.relabel_({"b", "c"});
        lan_guess.to_(Device.cuda);
        // lan_guess.print_diagram();
        // lan_guess.print_blocks();

        MyOp2 H = MyOp2(27);

        std::vector<UniTensor> eigs = linalg::Lanczos(
          &H, lan_guess, "Gnd", 9.999999999999999988e-15, 10000, 1, false, true, 0, true);
        cytnx_double ev = (cytnx_double)eigs[0].get_block_().item().real();
        std::cout << ev << ' ' << evans << std::endl;
        EXPECT_TRUE(std::fabs(ev - evans) < 1e-5);
        // EXPECT_DOUBLE_EQ(ev, evans);
      }

      // #943: H = diag(1, 2) on the GPU, acting on a rank-1 Tensor or a rank-1 dense UniTensor.
      class GpuDiagonalOp : public LinOp {
       public:
        GpuDiagonalOp() : LinOp("mv", 2, Type.Double, Device.cuda) {
          mat_ = zeros({2, 2}, Type.Double);
          mat_.at<cytnx_double>({0, 0}) = 1.0;
          mat_.at<cytnx_double>({1, 1}) = 2.0;
          mat_.to_(Device.cuda);
        }
        Tensor matvec(const Tensor& v) override { return linalg::Dot(mat_, v); }
        UniTensor matvec(const UniTensor& v) override {
          return UniTensor(linalg::Dot(mat_, v.get_block_()));
        }

       private:
        Tensor mat_;
      };

      Tensor GpuVector2(double x0, double x1) {
        Tensor v = zeros({2}, Type.Double);
        v.at<cytnx_double>({0}) = x0;
        v.at<cytnx_double>({1}) = x1;
        return v.to(Device.cuda);
      }

      // Expects the eigenpair (1, +-(1, 0)) of diag(1, 2).
      void ExpectGpuDiagonalGroundState(const Tensor& energy, const Tensor& vec) {
        EXPECT_EQ(vec.device(), Device.cuda);
        EXPECT_NEAR(double(energy.to(Device.cpu).item().real()), 1.0, 1e-12);
        Tensor v = vec.to(Device.cpu);
        EXPECT_NEAR(std::abs(v.at<cytnx_double>({0})), 1.0, 1e-12);
        EXPECT_NEAR(v.at<cytnx_double>({1}), 0.0, 1e-12);
      }

      // The start vector is the ground state itself, so the Krylov space is one-dimensional.
      TEST(LanczosGnd, GpuStartAtGroundState) {
        GpuDiagonalOp op;
        std::vector<Tensor> eigs =
          linalg::Lanczos(&op, GpuVector2(1.0, 0.0), "Gnd", 1.0e-14, 20, 1, true, false, 0, false);
        ASSERT_EQ(eigs.size(), 2);
        ExpectGpuDiagonalGroundState(eigs[0], eigs[1]);

        std::vector<UniTensor> eigs_ut = linalg::Lanczos(
          &op, UniTensor(GpuVector2(1.0, 0.0)), "Gnd", 1.0e-14, 20, 1, true, false, 0, false);
        ASSERT_EQ(eigs_ut.size(), 2);
        ExpectGpuDiagonalGroundState(eigs_ut[0].get_block_(), eigs_ut[1].get_block_());
      }

      // The Krylov space of (1, 1) is the whole two-dimensional space.
      TEST(LanczosGnd, GpuKrylovSpaceIsWholeSpace) {
        GpuDiagonalOp op;
        std::vector<Tensor> eigs =
          linalg::Lanczos(&op, GpuVector2(1.0, 1.0), "Gnd", 1.0e-300, 20, 1, true, false, 0, false);
        ASSERT_EQ(eigs.size(), 2);
        ExpectGpuDiagonalGroundState(eigs[0], eigs[1]);

        std::vector<UniTensor> eigs_ut = linalg::Lanczos(
          &op, UniTensor(GpuVector2(1.0, 1.0)), "Gnd", 1.0e-300, 20, 1, true, false, 0, false);
        ASSERT_EQ(eigs_ut.size(), 2);
        ExpectGpuDiagonalGroundState(eigs_ut[0].get_block_(), eigs_ut[1].get_block_());
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
