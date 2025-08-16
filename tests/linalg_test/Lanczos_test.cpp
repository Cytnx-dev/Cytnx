#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "cytnx.hpp"

using namespace cytnx;
using namespace testing;

namespace {
  // define the customize LinOp
  class MatOp : public LinOp {
   public:
    Tensor opMat;
    Tensor T_init;
    MatOp(const cytnx_uint64& nx = 1, const int& dtype = Type.Double);
    Tensor matvec(const Tensor& v) override { return (linalg::Dot(opMat, v)); }
    void InitVec();
  };
  MatOp::MatOp(const cytnx_uint64& in_nx, const int& in_dtype) : LinOp("mv", in_nx, in_dtype) {
    opMat = zeros({in_nx, in_nx}, this->dtype(), this->device());
    if (Type.is_float(this->dtype())) {
      random::Make_normal(opMat, 0.0, 1.0, 0);
    }
    opMat += opMat.permute({1, 0}).Conj();
    InitVec();
  }
  void MatOp::InitVec() {
    T_init = zeros(nx(), this->dtype());
    if (Type.is_float(this->dtype())) {
      random::Make_normal(T_init, 0.0, 1.0, 0);
    }
  }

  // the function to check the answer
  bool CheckResult(MatOp& H, const std::vector<Tensor>& lanczos_eigs, const std::string& which,
                   const cytnx_uint64 k);

  void ExcuteTest(const std::string& which, const int& mat_type = Type.Double,
                  const cytnx_uint64& k = 5, cytnx_uint64 dim = 23) {
    MatOp H = MatOp(dim, mat_type);
    const cytnx_uint64 maxiter = 10000;
    const cytnx_double cvg_crit = 0;
    std::vector<Tensor> lanczos_eigs = linalg::Lanczos(&H, H.T_init, which, maxiter, cvg_crit, k);
    bool is_pass = CheckResult(H, lanczos_eigs, which, k);
    EXPECT_TRUE(is_pass);
  }

  // error test
  class ErrorTestClass {
   public:
    std::string which = "LM";
    cytnx_uint64 k = 1;
    cytnx_uint64 maxiter = 10000;
    cytnx_double cvg_crit = 0;
    ErrorTestClass(){};
    cytnx_uint64 dim = 25;
    bool is_V = true;
    cytnx_int32 ncv = 0;
    void ExcuteErrorTest();
    // set
    void Set_mat_type(const int _mat_type) {
      mat_type = _mat_type;
      H = MatOp(dim, mat_type);
    }

   private:
    int mat_type = Type.Double;
    MatOp H = MatOp(dim, mat_type);
  };
  void ErrorTestClass::ExcuteErrorTest() {
    try {
      auto lanczos_eigs = linalg::Lanczos(&H, H.T_init, which, maxiter, cvg_crit, k, is_V, ncv);
      FAIL();
    } catch (const std::exception& ex) {
      auto err_msg = ex.what();
      std::cerr << err_msg << std::endl;
      SUCCEED();
    }
  }

  // For given 'which' = 'LM', 'SM', ...etc, sort the given eigenvalues.
  bool cmpNorm(const Scalar& l, const Scalar& r) { return abs(l) < abs(r); }
  bool cmpAlgebra(const Scalar& l, const Scalar& r) { return l < r; }
  std::vector<Scalar> OrderEigvals(const Tensor& eigvals, const std::string& order_type) {
    char small_or_large = order_type[0];  //'S' or 'L'
    char metric_type = order_type[1];  //'M', o 'A'
    auto eigvals_len = eigvals.shape()[0];
    auto ordered_eigvals = std::vector<Scalar>(eigvals_len, Scalar());
    for (cytnx_uint64 i = 0; i < eigvals_len; ++i) ordered_eigvals[i] = eigvals.storage().at(i);
    std::function<bool(const Scalar&, const Scalar&)> cmpFncPtr;
    if (metric_type == 'M') {
      cmpFncPtr = cmpNorm;
    } else if (metric_type == 'A') {
      cmpFncPtr = cmpAlgebra;
    } else {
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

  // compare the lanczos results with full spectrum (calculated by the function Eig.)
  bool CheckResult(MatOp& H, const std::vector<Tensor>& lanczos_eigs, const std::string& which,
                   const cytnx_uint64 k) {
    // get full spectrum (eigenvalues)
    std::vector<Tensor> full_eigs = linalg::Eigh(H.opMat);
    Tensor full_eigvals = full_eigs[0];
    std::vector<Scalar> ordered_eigvals = OrderEigvals(full_eigvals, which);
    auto fst_few_eigvals =
      std::vector<Scalar>(ordered_eigvals.begin(), ordered_eigvals.begin() + k);
    Tensor lanczos_eigvals = lanczos_eigs[0];
    Tensor lanczos_eigvecs = lanczos_eigs[1];
    auto dtype = H.dtype();
    const double tolerance = (dtype == Type.ComplexFloat || dtype == Type.Float) ? 1.0e-4 : 1.0e-12;
    // check the number of the eigenvalues
    cytnx_uint64 lanczos_eigvals_len = lanczos_eigvals.shape()[0];
    if (lanczos_eigvals_len != k) return false;
    for (cytnx_uint64 i = 0; i < k; ++i) {
      auto lanczos_eigval = lanczos_eigvals.storage().at(i);
      // if k == 1, lanczos_eigvecs will be a rank-1 tensor
      auto lanczos_eigvec = k == 1 ? lanczos_eigvecs : lanczos_eigvecs(i);
      auto exact_eigval = fst_few_eigvals[i];
      // check eigen value by comparing with the full spectrum results.
      // avoid, for example, lanczos_eigval = 1 + 3j, exact_eigval = 1 - 3j, which = 'SM'
      // check the is the eigenvector correct
      auto eigval_err = abs(abs(lanczos_eigval) - abs(exact_eigval)) / abs(exact_eigval);
      // std::cout << "eigval err=" << eigval_err << std::endl;
      if (eigval_err >= tolerance) return false;
      // check the is the eigenvector correct
      auto resi_err = GetResidue(H, lanczos_eigval, lanczos_eigvec);
      // std::cout << "resi err=" << resi_err << std::endl;
      if (resi_err >= tolerance) return false;
      // check phase
    }
    return true;
  }
}  // namespace

// corrected test
// 1-1, test for 'which' = 'LM'
TEST(Lanczos, which_LM_test) {
  std::string which = "LM";
  ExcuteTest(which);
}

// 1-2, test for 'which' = 'LR'
TEST(Lanczos, which_LA_test) {
  std::string which = "LA";
  ExcuteTest(which);
}

// 1-3, test for 'which' = 'SA'
TEST(Lanczos, which_SR_test) {
  std::string which = "SA";
  ExcuteTest(which);
}

// 1-4, test matrix is all type
TEST(Lanczos, mat_type_all_test) {
  std::string which = "LM";
  std::vector<int> dtypes = {Type.ComplexDouble, Type.ComplexFloat, Type.Double, Type.Float};
  for (auto dtype : dtypes) {
	std::cout << "input which" << which << std::endl;
    ExcuteTest(which, dtype);
  }
}

// 1-8, test eigenalue number k = 1
TEST(Lanczos, k1_test) {
  std::string which = "LM";
  auto mat_type = Type.Double;
  cytnx_uint64 k = 1;
  ExcuteTest(which, mat_type, k);
}

// 1-9, test eigenalue number k match maximum, that means k = dim.
TEST(Lanczos, k_large) {
  std::string which = "LM";
  auto mat_type = Type.Double;
  cytnx_uint64 k, dim;
  dim = 13;
  k = 11;
  ExcuteTest(which, mat_type, k, dim);
}

// 1-10, test the smallest matrix dimenstion.
TEST(Lanczos, smallest_dim) {
  std::string which = "LM";
  auto mat_type = Type.Double;
  cytnx_uint64 k, dim;
  k = 1;
  dim = 3;
  ExcuteTest(which, mat_type, k, dim);
}

// 1-11, test 'is_V' is false
// 1-12, test 'v_bose' is true
// 1-13, test converge criteria is large such that the iteration time may not reach 'k'

// 2-1, test for wrong input 'which'
TEST(Lanczos, err_which) {
  ErrorTestClass err_task;
  err_task.which = "ML";
  err_task.ExcuteErrorTest();
}

// 2-2, test for wrong input LinOp dtype
TEST(Lanczos, err_mat_type) {
  ErrorTestClass err_task;
  err_task.Set_mat_type(Type.Int64);
  err_task.ExcuteErrorTest();
}

// 2-3, test for 'k' = 0
TEST(Lanczos, err_zero_k) {
  ErrorTestClass err_task;
  err_task.k = 0;
  err_task.ExcuteErrorTest();
}

// 2-4, test for 'k' > 'max_iter'
TEST(Lanczos, err_k_too_large) {
  ErrorTestClass err_task;
  err_task.k = err_task.dim + 1;
  err_task.ExcuteErrorTest();
}

// 2-5, test cvg_crit <= 0
TEST(Lanczos, err_crit_negative) {
  ErrorTestClass err_task;
  err_task.cvg_crit = -0.001;
  err_task.ExcuteErrorTest();
}

// 2-6, test ncv is out of allowd range
TEST(Lanczos, err_ncv_out_of_range) {
  ErrorTestClass err_task;
  err_task.ncv = err_task.k + 1;
  err_task.ExcuteErrorTest();
  err_task.ncv = err_task.dim + 1;
  err_task.ExcuteErrorTest();
}
