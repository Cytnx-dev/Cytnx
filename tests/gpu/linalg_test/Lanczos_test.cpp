#include <gtest/gtest.h>

#include "../test_tools.h"
#include "cytnx.hpp"
#include "linalg_test.h"

namespace cytnx {
  namespace gpu_test {
    namespace {
      using gpu_test::ferm_dense_lowest;
      using gpu_test::FermiAdaOp;
      using gpu_test::make_ferm_ada_ket;

      // define the customize LinOp
      class GpuLanczosMatOp : public LinOp {
       public:
        Tensor opMat;
        Tensor T_init;
        GpuLanczosMatOp(const cytnx_uint64& nx, const int& dtype, const int& in_device);
        Tensor matvec(const Tensor& v) override { return (linalg::Dot(opMat, v)); }
        void InitVec();
        friend class CheckOp;
      };
      GpuLanczosMatOp::GpuLanczosMatOp(const cytnx_uint64& in_nx, const int& in_dtype,
                                       const int& in_device)
          : LinOp("mv", in_nx, in_dtype, in_device) {
        opMat = zeros({in_nx, in_nx}, this->dtype(), this->device());
        if (Type.is_float(this->dtype())) {
          random::normal_(opMat, 0.0, 1.0, 0);
        }
        opMat += opMat.permute({1, 0}).Conj();  // Hermitian
        InitVec();
      }
      void GpuLanczosMatOp::InitVec() {
        T_init = zeros({nx()}, this->dtype(), this->device());
        if (Type.is_float(this->dtype())) {
          random::normal_(T_init, 0.0, 1.0, 0);
        }
      }

      class CheckOp : public LinOp {
       public:
        Tensor opMat;
        Tensor T_init;
        GpuLanczosMatOp* op;
        CheckOp(GpuLanczosMatOp* in_op)
            : op(in_op), LinOp("mv", in_op->nx(), in_op->dtype(), Device.cpu) {
          opMat = op->opMat.to(Device.cpu);
          T_init = op->T_init.to(Device.cpu);
        }
        Tensor matvec(const Tensor& v) override { return (linalg::Dot(opMat, v)); }
      };

      // the function to check the answer
      static bool CheckResult(CheckOp& H, const std::vector<Tensor>& lanczos_eigs_cuda,
                              const std::vector<Tensor>& lanczos_eigs_cpu);

      static void ExcuteTest(const std::string& which, const int& mat_type = Type.Double,
                             const cytnx_uint64& k = 5, cytnx_uint64 dim = 23) {
        GpuLanczosMatOp H = GpuLanczosMatOp(dim, mat_type, Device.cuda);
        CheckOp H_check = CheckOp(&H);
        const cytnx_uint64 maxiter = 10000;
        auto dtype = H.dtype();
        const cytnx_double cvg_crit = 0;
        std::vector<Tensor> lanczos_eigs_cuda =
          linalg::Lanczos(&H, H.T_init, which, maxiter, cvg_crit, k);
        for (auto& lanczos_eig : lanczos_eigs_cuda) {
          EXPECT_EQ(lanczos_eig.device(), Device.cuda);
        }
        std::vector<Tensor> lanczos_eigs_cpu =
          linalg::Lanczos(&H_check, H_check.T_init, which, maxiter, cvg_crit, k);
        bool is_pass = CheckResult(H_check, lanczos_eigs_cuda, lanczos_eigs_cpu);
        EXPECT_TRUE(is_pass);
      }

      // get resigue |Hv - ev|
      static Scalar GetResidue(CheckOp& H, const Scalar& eigval, const Tensor& eigvec) {
        Tensor resi_vec = H.matvec(eigvec) - eigval * eigvec;
        Scalar resi = resi_vec.Norm().item();
        return resi;
      }

      static bool CheckResult(CheckOp& H, const std::vector<Tensor>& lanczos_eigs_cuda,
                              const std::vector<Tensor>& lanczos_eigs_cpu) {
        auto dtype = H.dtype();
        const double tolerance =
          (dtype == Type.ComplexFloat || dtype == Type.Float) ? 1.0e-4 : 1.0e-4;

        // Check eigenvalues copmared with the results from cpu.
        if (lanczos_eigs_cuda.size() != lanczos_eigs_cpu.size()) {
          return false;
        }
        Tensor eigval_cuda_to_cpu = lanczos_eigs_cuda[0].to(Device.cpu);
        Tensor eigval_cpu = lanczos_eigs_cpu[0];

        bool is_same_eigval = AreNearlyEqTensor(eigval_cuda_to_cpu, eigval_cpu, tolerance);
        if (!is_same_eigval) {
          return false;
        }

        // Check eigenvectors. We have not compare the eigenvector directly since they may have
        // different phase.
        Tensor lanczos_eigvecs = lanczos_eigs_cuda[1].to(Device.cpu);

        // check the number of the eigenvalues
        int k = eigval_cpu.shape()[0];
        for (cytnx_uint64 i = 0; i < k; ++i) {
          // if k == 1, lanczos_eigvecs will be a rank-1 tensor
          auto lanczos_eigvec = k == 1 ? lanczos_eigvecs : lanczos_eigvecs(i);
          auto lanczos_eigval = eigval_cuda_to_cpu.at({i});
          // check the is the eigenvector correct
          auto resi_err = GetResidue(H, lanczos_eigval, lanczos_eigvec);
          if (resi_err >= tolerance) return false;
        }
        return true;
      }

      // corrected test
      // 1-1, test for 'which' = 'LM'
      TEST(Lanczos, GpuWhichLMTest) {
        std::string which = "LM";
        ExcuteTest(which);
      }

      // 1-2, test for 'which' = 'LR'
      TEST(Lanczos, GpuWhichLATest) {
        std::string which = "LA";
        ExcuteTest(which);
      }

      // 1-3, test for 'which' = 'SA'
      TEST(Lanczos, GpuWhichSRTest) {
        std::string which = "SA";
        ExcuteTest(which);
      }

      // 1-4, test matrix is all type
      TEST(Lanczos, GpuMatTypeAllTest) {
        std::string which = "LM";
        std::vector<int> dtypes = {Type.ComplexDouble, Type.ComplexFloat, Type.Double, Type.Float};
        for (auto dtype : dtypes) {
          ExcuteTest(which, dtype);
        }
      }

      // 1-8, test eigenalue number k = 1
      TEST(Lanczos, GpuK1Test) {
        std::string which = "LM";
        auto mat_type = Type.Double;
        cytnx_uint64 k = 1;
        ExcuteTest(which, mat_type, k);
      }

      // 1-9, test eigenalue number k match maximum, that means k = dim.
      TEST(Lanczos, GpuKLarge) {
        std::string which = "LM";
        auto mat_type = Type.Double;
        cytnx_uint64 k, dim;
        dim = 13;
        k = 11;
        ExcuteTest(which, mat_type, k, dim);
      }

      // 1-10, test the smallest matrix dimenstion.
      TEST(Lanczos, GpuSmallestDim) {
        std::string which = "LM";
        auto mat_type = Type.Double;
        cytnx_uint64 k, dim;
        k = 1;
        dim = 3;
        ExcuteTest(which, mat_type, k, dim);
      }

      // GPU fermionic Krylov: O = A^dag A (sign-flip-active 4-leg A); eigenpairs computed on the
      // GPU and checked against an independent CPU dense diagonalization.
      TEST(Lanczos, GpuFermionicLanczosArpack) {  // ARPACK symmetric Lanczos 'SA', two lowest
        const double tol = 1e-7;
        UniTensor A = make_ferm_A();
        UniTensor v0 = make_ferm_ada_ket(A);
        auto low = ferm_dense_lowest(A, 2);
        FermiAdaOp op(A.to(Device.cuda), ferm_ket_nx(v0));
        // exact argument types select the ARPACK which-based Lanczos overload over the method-based
        // one (uint64 maxiter, double cvg_crit, int32 ncv), so the call resolves without a cast.
        auto eigs = to_cpu(linalg::Lanczos(&op, v0.to(Device.cuda), std::string("SA"),
                                           (cytnx_uint64)1000, (cytnx_double)1e-12, (cytnx_uint64)2,
                                           true, (cytnx_int32)0, false));
        expect_lowest_states(A, eigs, low, tol);
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
