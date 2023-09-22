#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "cytnx.hpp"

using namespace cytnx;
using namespace testing;

namespace ArnoldiTest {

  cytnx_double tolerance = 1.0e-9;

  // define the customize LinOp
  class MatOp : public LinOp {
   public:
    Tensor opMat;
    Tensor T_init;
    MatOp(const cytnx_uint64& nx = 1, const int& dtype = Type.Double);
    Tensor matvec(const Tensor& v) override { return (linalg::Dot(opMat, v)); }
    void InitVec();
  };
  MatOp::MatOp(const cytnx_uint64& in_nx, const int& in_dtype)
      : LinOp("mv", in_nx, in_dtype, Device.cuda) {
    opMat = zeros({in_nx, in_nx}, this->dtype(), this->device());
    if (Type.is_float(this->dtype())) {
      random::Make_normal(opMat, 0.0, 1.0, 0);
    }
    InitVec();
  }
  void MatOp::InitVec() {
    T_init = zeros(nx(), this->dtype()).to(this->device());
    if (Type.is_float(this->dtype())) {
      random::Make_normal(T_init, 0.0, 1.0, 0);
    }
  }

  // the function to check the answer
  bool CheckResult(MatOp& H, const std::vector<Tensor>& arnoldi_eigs, const std::string& which,
                   const cytnx_uint64 k);

  void ExcuteTest(const std::string& which, const int& mat_type = Type.ComplexDouble,
                  const cytnx_uint64& k = 5, cytnx_uint64 dim = 23) {
    MatOp H = MatOp(dim, mat_type);
    const cytnx_uint64 maxiter = 10000;
    const cytnx_double cvg_crit = tolerance;
    std::vector<Tensor> arnoldi_eigs = linalg::Arnoldi(&H, H.T_init, which, maxiter, cvg_crit, k);
    bool is_pass = CheckResult(H, arnoldi_eigs, which, k);
    EXPECT_TRUE(is_pass);
  }

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

  // 1-4, test for 'which' = 'SM'
  TEST(Arnoldi, gpu_which_SM_test) {
    std::string which = "SM";
    ExcuteTest(which);
  }

  // 1-5, test for 'which' = 'SR'
  TEST(Arnoldi, gpu_which_SR_test) {
    std::string which = "SR";
    ExcuteTest(which);
  }

  // 1-6, test for 'which' = 'SI'
  TEST(Arnoldi, gpu_which_SI_test) {
    std::string which = "SI";
    ExcuteTest(which);
  }

  // 1-7, test matrix is real type
  TEST(Arnoldi, gpu_mat_type_real_test) {
    std::string which = "LM";
    auto mat_type = Type.Double;
    ExcuteTest(which, mat_type);
  }

  // 1-8, test eigenalue number k = 1
  TEST(Arnoldi, gpu_k1_test) {
    std::string which = "LM";
    auto mat_type = Type.ComplexDouble;
    cytnx_uint64 k = 1;
    ExcuteTest(which, mat_type, k);
  }

  // 1-9, test eigenalue number k match maximum, that means k = dim.
  TEST(Arnoldi, gpu_k_max) {
    std::string which = "LM";
    auto mat_type = Type.ComplexDouble;
    cytnx_uint64 k, dim;
    k = dim = 13;
    ExcuteTest(which, mat_type, k, dim);
  }

  // 1-10, test the smallest matrix dimenstion.
  TEST(Arnoldi, gpu_smallest_dim) {
    std::string which = "LM";
    auto mat_type = Type.ComplexDouble;
    cytnx_uint64 k, dim;
    k = 1;
    dim = 3;
    ExcuteTest(which, mat_type, k, dim);
  }

  // 1-11, test 'is_V' is false
  // 1-12, test 'v_bose' is true
  // 1-13, test converge criteria is large such that the iteration time may not reach 'k'

  // error test
  class ErrorTestClass {
   public:
    std::string which = "LM";
    cytnx_uint64 k = 1;
    cytnx_uint64 maxiter = 10000;
    cytnx_double cvg_crit = tolerance;
    ErrorTestClass(){};
    void ExcuteErrorTest();
    // set
    void Set_dim(const cytnx_uint64 _dim) {
      dim = _dim;
      H = MatOp(dim, mat_type);
    }
    void Set_mat_type(const int _mat_type) {
      mat_type = _mat_type;
      H = MatOp(dim, mat_type);
    }

   private:
    int mat_type = Type.ComplexDouble;
    cytnx_uint64 dim = 3;
    MatOp H = MatOp(dim, mat_type);
  };
  void ErrorTestClass::ExcuteErrorTest() {
    try {
      auto arnoldi_eigs = linalg::Arnoldi(&H, H.T_init, which, maxiter, cvg_crit, k);
      FAIL();
    } catch (const std::exception& ex) {
      auto err_msg = ex.what();
      std::cerr << err_msg << std::endl;
      SUCCEED();
    }
  }

  // 2-1, test for wrong input 'which'
  TEST(Arnoldi, gpu_err_which) {
    ErrorTestClass err_task;
    err_task.which = "ML";
    err_task.ExcuteErrorTest();
  }

  // 2-2, test for wrong input LinOp dtype
  TEST(Arnoldi, gpu_err_mat_type) {
    ErrorTestClass err_task;
    err_task.Set_mat_type(Type.Int64);
    err_task.ExcuteErrorTest();
  }

  // 2-3, test for 'k' = 0
  TEST(Arnoldi, gpu_err_zero_k) {
    ErrorTestClass err_task;
    err_task.k = 0;
    err_task.ExcuteErrorTest();
  }

  // 2-4, test for 'k' > 'max_iter'
  TEST(Arnoldi, gpu_err_k_too_large) {
    ErrorTestClass err_task;
    err_task.k = 0;
    err_task.ExcuteErrorTest();
  }

  // 2-5, test cvg_crit <= 0
  TEST(Arnoldi, gpu_err_crit_negative) {
    ErrorTestClass err_task;
    err_task.cvg_crit = -0.001;
    err_task.ExcuteErrorTest();
  }

  // For given 'which' = 'LM', 'SM', ...etc, sort the given eigenvalues.
  bool cmpNorm(const Scalar& l, const Scalar& r) { return abs(l) < abs(r); }
  bool cmpReal(const Scalar& l, const Scalar& r) { return l.real() < r.real(); }
  bool cmpImag(const Scalar& l, const Scalar& r) { return l.imag() < r.imag(); }
  std::vector<Scalar> OrderEigvals(const Tensor& eigvals, const std::string& order_type) {
    char small_or_large = order_type[0];  //'S' or 'L'
    char metric_type = order_type[1];  //'M', 'R' or 'I'
    auto eigvals_len = eigvals.shape()[0];
    auto ordered_eigvals = std::vector<Scalar>(eigvals_len, Scalar());
    for (cytnx_uint64 i = 0; i < eigvals_len; ++i) ordered_eigvals[i] = eigvals.storage().at(i);
    std::function<bool(const Scalar&, const Scalar&)> cmpFncPtr;
    if (metric_type == 'M') {
      cmpFncPtr = cmpNorm;
    } else if (metric_type == 'R') {
      cmpFncPtr = cmpReal;
    } else if (metric_type == 'I') {
      cmpFncPtr = cmpImag;
    } else {  // wrong input
      ;
    }
    // sort eigenvalues
    if (small_or_large == 'S') {
      std::stable_sort(ordered_eigvals.begin(), ordered_eigvals.end(), cmpFncPtr);
    } else {  // 'L'
      std::stable_sort(ordered_eigvals.rbegin(), ordered_eigvals.rend(), cmpFncPtr);
    }
    return ordered_eigvals;
  }

  // get resigue |Hv - ev|
  Scalar GetResidue(MatOp& H, const Scalar& eigval, const Tensor& eigvec) {
    Tensor resi_vec = H.matvec(eigvec) - eigval * eigvec;
    Scalar resi = resi_vec.Norm().item();
    return resi;
  }

  // compare the arnoldi results with full spectrum (calculated by the function Eig.)
  bool CheckResult(MatOp& H, const std::vector<Tensor>& arnoldi_eigs, const std::string& which,
                   const cytnx_uint64 k) {
    // get full spectrum (eigenvalues)
    std::vector<Tensor> full_eigs = linalg::Eig(H.opMat);
    Tensor full_eigvals = full_eigs[0];
    std::vector<Scalar> ordered_eigvals = OrderEigvals(full_eigvals, which);
    auto fst_few_eigvals =
      std::vector<Scalar>(ordered_eigvals.begin(), ordered_eigvals.begin() + k);
    Tensor arnoldi_eigvals = arnoldi_eigs[0];
    Tensor arnoldi_eigvecs = arnoldi_eigs[1];

    // check the number of the eigenvalues
    cytnx_uint64 arnoldi_eigvals_len = arnoldi_eigvals.shape()[0];
    if (arnoldi_eigvals_len != k) return false;
    for (cytnx_uint64 i = 0; i < k; ++i) {
      auto arnoldi_eigval = arnoldi_eigvals.storage().at(i);
      // if k == 1, arnoldi_eigvecs will be a rank-1 tensor
      auto arnoldi_eigvec = k == 1 ? arnoldi_eigvecs : arnoldi_eigvecs(i);
      auto exact_eigval = fst_few_eigvals[i];
      // check eigen value by comparing with the full spectrum results.
      // avoid, for example, arnoldi_eigval = 1 + 3j, exact_eigval = 1 - 3j, which = 'SM'
      auto eigval_err = abs(arnoldi_eigval) - abs(exact_eigval);
      if (eigval_err >= tolerance) return false;
      // check the is the eigenvector correct
      auto resi_err = GetResidue(H, arnoldi_eigval, arnoldi_eigvec);
      if (resi_err >= tolerance) return false;
      // check phase
    }
    return true;
  }

}  // namespace ArnoldiTest
