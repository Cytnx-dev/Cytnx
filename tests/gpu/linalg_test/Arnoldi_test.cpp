#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "cytnx.hpp"
#include "../test_tools.h"

using namespace cytnx;
using namespace testing;

namespace {

  // define the customize LinOp
  class MatOp : public LinOp {
   public:
    Tensor opMat;
    Tensor T_init;
    MatOp(const cytnx_uint64& nx, const int& dtype, const int& in_device);
    Tensor matvec(const Tensor& v) override { return (linalg::Dot(opMat, v)); }
    void InitVec();
    friend class CheckOp;
  };
  MatOp::MatOp(const cytnx_uint64& in_nx, const int& in_dtype, const int& in_device)
      : LinOp("mv", in_nx, in_dtype, in_device) {
    opMat = zeros({in_nx, in_nx}, this->dtype(), this->device());
    if (Type.is_float(this->dtype())) {
      random::normal_(opMat, 0.0, 1.0, 0);
    }
    InitVec();
  }
  void MatOp::InitVec() {
    T_init = zeros(nx(), this->dtype(), this->device());
    if (Type.is_float(this->dtype())) {
      random::normal_(T_init, 0.0, 1.0, 0);
    }
  }

  class CheckOp : public LinOp {
   public:
    Tensor opMat;
    Tensor T_init;
    MatOp* op;
    CheckOp(MatOp* in_op) : op(in_op), LinOp("mv", in_op->nx(), in_op->dtype(), Device.cpu) {
      opMat = op->opMat.to(Device.cpu);
      T_init = op->T_init.to(Device.cpu);
    }
    Tensor matvec(const Tensor& v) override { return (linalg::Dot(opMat, v)); }
  };

  // the function to check the answer
  bool CheckResult(CheckOp& H, const std::vector<Tensor>& arnoldi_eigs_cuda,
                   const std::vector<Tensor>& arnoldi_eigs_cpu);

  void ExcuteTest(const std::string& which, const int& mat_type = Type.ComplexDouble,
                  const cytnx_uint64& k = 5, cytnx_uint64 dim = 23) {
    MatOp H = MatOp(dim, mat_type, Device.cuda);
    CheckOp H_check = CheckOp(&H);
    const cytnx_uint64 maxiter = 10000;
    auto dtype = H.dtype();
    const cytnx_double cvg_crit = 0;
    std::vector<Tensor> arnoldi_eigs_cuda =
      linalg::Arnoldi(&H, H.T_init, which, maxiter, cvg_crit, k);
    for (auto& arnoldi_eig : arnoldi_eigs_cuda) {
      EXPECT_EQ(arnoldi_eig.device(), Device.cuda);
    }
    std::vector<Tensor> arnoldi_eigs_cpu =
      linalg::Arnoldi(&H_check, H_check.T_init, which, maxiter, cvg_crit, k);
    bool is_pass = CheckResult(H_check, arnoldi_eigs_cuda, arnoldi_eigs_cpu);
    EXPECT_TRUE(is_pass);
  }

  // get resigue |Hv - ev|
  Scalar GetResidue(CheckOp& H, const Scalar& eigval, const Tensor& eigvec) {
    Tensor resi_vec = H.matvec(eigvec) - eigval * eigvec;
    Scalar resi = resi_vec.Norm().item();
    return resi;
  }

  bool CheckResult(CheckOp& H, const std::vector<Tensor>& arnoldi_eigs_cuda,
                   const std::vector<Tensor>& arnoldi_eigs_cpu) {
    auto dtype = H.dtype();
    const double tolerance = (dtype == Type.ComplexFloat || dtype == Type.Float) ? 1.0e-4 : 1.0e-12;

    // Check eigenvalues copmared with the results from cpu.
    if (arnoldi_eigs_cuda.size() != arnoldi_eigs_cpu.size()) {
      return false;
    }
    Tensor eigval_cuda_to_cpu = arnoldi_eigs_cuda[0].to(Device.cpu);
    Tensor eigval_cpu = arnoldi_eigs_cpu[0];
    bool is_same_eigval = TestTools::AreNearlyEqTensor(eigval_cuda_to_cpu, eigval_cpu, tolerance);
    if (!is_same_eigval) {
      return false;
    }

    // Check eigenvectors. We have not compare the eigenvector directly since they may have
    // different phase.
    Tensor arnoldi_eigvecs = arnoldi_eigs_cuda[1].to(Device.cpu);

    // check the number of the eigenvalues
    int k = eigval_cpu.shape()[0];
    for (cytnx_uint64 i = 0; i < k; ++i) {
      // if k == 1, arnoldi_eigvecs will be a rank-1 tensor
      auto arnoldi_eigvec = k == 1 ? arnoldi_eigvecs : arnoldi_eigvecs(i);
      auto arnoldi_eigval = eigval_cuda_to_cpu.at({i});
      // check the is the eigenvector correct
      auto resi_err = GetResidue(H, arnoldi_eigval, arnoldi_eigvec);
      // std::cout << "resi err=" << resi_err << std::endl;
      if (resi_err >= tolerance) return false;
    }
    return true;
  }
}  // namespace

// corrected test
// 1-1, test for 'which' = 'LM'
TEST(Arnoldi, gpu_which_LM_test) {
  std::string which = "LM";
  ExcuteTest(which);
}

// 1-2, test for 'which' = 'LR'
TEST(Arnoldi, gpu_which_LR_test) {
  std::string which = "LR";
  ExcuteTest(which);
}

// 1-3, test for 'which' = 'LI'
TEST(Arnoldi, gpu_which_LI_test) {
  std::string which = "LI";
  ExcuteTest(which);
}

// 1-4, test for 'which' = 'SR'
TEST(Arnoldi, gpu_which_SR_test) {
  std::string which = "SR";
  ExcuteTest(which);
}

// 1-5, test for 'which' = 'SI'
TEST(Arnoldi, gpu_which_SI_test) {
  std::string which = "SI";
  ExcuteTest(which);
}

// 1-6, test matrix is all type
TEST(Arnoldi, gpu_mat_type_all_test) {
  std::string which = "LM";
  std::vector<int> dtypes = {Type.ComplexDouble, Type.ComplexFloat, Type.Double, Type.Float};
  for (auto dtype : dtypes) {
    ExcuteTest(which, dtype);
  }
}

// 1-7, test eigenalue number k = 1
TEST(Arnoldi, gpu_k1_test) {
  std::string which = "LM";
  auto mat_type = Type.ComplexDouble;
  cytnx_uint64 k = 1;
  ExcuteTest(which, mat_type, k);
}

// 1-8, test eigenalue number close to dim.
TEST(Arnoldi, gpu_k_large) {
  std::string which = "LM";
  auto mat_type = Type.ComplexDouble;
  cytnx_uint64 k, dim;
  dim = 13;
  k = 11;
  ExcuteTest(which, mat_type, k, dim);
}

// 1-9, test the smallest matrix dimenstion.
TEST(Arnoldi, gpu_smallest_dim) {
  std::string which = "LM";
  auto mat_type = Type.ComplexDouble;
  cytnx_uint64 k, dim;
  k = 1;
  dim = 3;
  ExcuteTest(which, mat_type, k, dim);
}
