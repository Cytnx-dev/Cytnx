#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "cytnx.hpp"

using namespace cytnx;
using namespace testing;

namespace {

  /*
   *   "al"+---A--- "ar"               +-"al"
   *       |   |                       |
   *      (l)  |"phys"    = lambda_i  (l)
   *       |   |                       |
   *   "bl"+---B--- "br"               +-"bl"
   */
  // define the Transfer matrix LinOp
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

    // only for test
    /*
     *   "al"+---A--- "ar"
     *           |
     *           |"phys"
     *           |
     *   "bl"+---B--- "br"
     */
    Tensor GetOpAsMat() {
      int D = A.shape()[0];
      auto tmp = Contract(A, B);
      auto mat = tmp.permute({"al", "bl", "ar", "br"}, 2).get_block_().reshape(D * D, D * D);
      return mat;
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

  class MyOp2 : public LinOp {
   public:
    UniTensor H;
    MyOp2(int dim) : LinOp("mv", dim) {
      Bond lan_I = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
      Bond lan_J = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
      H = UniTensor({lan_I, lan_J});
      double low = -1.0, high = 1.0;
      int seed = 0;
      H.uniform_(low, high, seed);
      H.set_labels({"a", "b"});
      // H.print_diagram();
      // H.print_blocks();
    }
    UniTensor matvec(const UniTensor& psi) override {
      auto out = (H.astype(psi.dtype())).contract(psi);
      out.set_labels({"b", "c"});
      return out;
    }
  };

  // For given 'which' = 'LM', 'SR', ...etc, sort the given eigenvalues.
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
  Scalar GetResidue(TMOp& H, const Scalar& eigval, const UniTensor& eigvec) {
    UniTensor resi_vec = H.matvec(eigvec) - eigval * eigvec;
    Scalar resi = resi_vec.Norm().item();
    return resi;
  }

  // compare the arnoldi results with full spectrum (calculated by the function Eig.)
  bool CheckResult(TMOp& H, const std::vector<UniTensor>& arnoldi_eigs, const std::string& which,
                   const cytnx_uint64 k) {
    // get full spectrum (eigenvalues)
    std::vector<Tensor> full_eigs = linalg::Eig(H.GetOpAsMat());
    Tensor full_eigvals = full_eigs[0];
    std::vector<Scalar> ordered_eigvals = OrderEigvals(full_eigvals, which);
    auto fst_few_eigvals =
      std::vector<Scalar>(ordered_eigvals.begin(), ordered_eigvals.begin() + k);
    Tensor arnoldi_eigvals = arnoldi_eigs[0].get_block_();

    // check the number of the eigenvalues
    cytnx_uint64 arnoldi_eigvals_len = arnoldi_eigvals.shape()[0];
    auto dtype = H.dtype();
    const double tolerance = (dtype == Type.ComplexFloat || dtype == Type.Float) ? 1.0e-4 : 1.0e-12;
    if (arnoldi_eigvals_len != k) return false;
    for (cytnx_uint64 i = 0; i < k; ++i) {
      auto arnoldi_eigval = arnoldi_eigvals.storage().at(i);
      // if k == 1, arnoldi_eigvecs will be a rank-1 tensor
      auto arnoldi_eigvec = arnoldi_eigs[i + 1];
      auto exact_eigval = fst_few_eigvals[i];
      // check eigen value by comparing with the full spectrum results.
      // avoid, for example, arnoldi_eigval = 1 + 3j, exact_eigval = 1 - 3j, which = 'LM'
      auto eigval_err = abs(abs(arnoldi_eigval) - abs(exact_eigval)) / abs(exact_eigval);
      // std::cout << "eigval err" << eigval_err << std::endl;
      if (eigval_err >= tolerance) return false;
      // check the is the eigenvector correct
      auto resi_err = GetResidue(H, arnoldi_eigval, arnoldi_eigvec);
      // std::cout << "resi err" << resi_err << std::endl;
      if (resi_err >= tolerance) return false;
      // check phase
    }
    return true;
  }

  void ExcuteTest(const std::string& which, const int& mat_type = Type.ComplexDouble,
                  const cytnx_uint64& k = 3) {
    int D = 5, d = 2;
    int dim = D * D;
    TMOp H = TMOp(d, D, dim, mat_type);
    const cytnx_uint64 maxiter = 10000;
    const cytnx_double cvg_crit = 0;
    std::vector<UniTensor> arnoldi_eigs =
      linalg::Arnoldi(&H, H.T_init, which, maxiter, cvg_crit, k);
    H.GetOpAsMat();
    bool is_pass = CheckResult(H, arnoldi_eigs, which, k);
    EXPECT_TRUE(is_pass);
  }
}  // namespace

// corrected test
// 1-1, test for 'which' = 'LM'
TEST(Arnoldi_Ut, which_LM_test) {
  std::string which = "LM";
  ExcuteTest(which);
}

// 1-2, test for 'which' = 'LR'
TEST(Arnoldi_Ut, which_LR_test) {
  std::string which = "LR";
  ExcuteTest(which);
}

// 1-3, test for 'which' = 'LI'
TEST(Arnoldi_Ut, which_LI_test) {
  std::string which = "LI";
  ExcuteTest(which);
}

// 1-4, test for 'which' = 'SR'
TEST(Arnoldi_Ut, which_SR_test) {
  std::string which = "SR";
  ExcuteTest(which);
}

// 1-5, test for 'which' = 'SI'
TEST(Arnoldi_Ut, which_SI_test) {
  std::string which = "SI";
  ExcuteTest(which);
}

// 1-6, test matrix is all allowed data type
TEST(Arnoldi_Ut, all_dtype_test) {
  std::string which = "LM";
  std::vector<int> dtypes = {Type.ComplexDouble, Type.ComplexFloat, Type.Double, Type.Float};
  for (auto dtype : dtypes) {
    ExcuteTest(which, dtype);
  }
}

// 1-7, test eigenalue number k = 1
TEST(Arnoldi_Ut, k1_test) {
  std::string which = "LM";
  auto mat_type = Type.ComplexDouble;
  cytnx_uint64 k = 1;
  ExcuteTest(which, mat_type, k);
}

// 1-8, test eigenalue number k is closed to dimension.
TEST(Arnoldi_Ut, k_large) {
  std::string which = "LM";
  auto mat_type = Type.ComplexDouble;
  cytnx_uint64 k;
  k = 23;  // dim = 25
  ExcuteTest(which, mat_type, k);
}

// 1-9, test the smallest matrix dimenstion.
TEST(Arnoldi_Ut, smallest_dim) {
  std::string which = "LM";
  auto mat_type = Type.ComplexDouble;
  cytnx_uint64 k;
  k = 1;
  ExcuteTest(which, mat_type, k);
}

// 1-10, test 'is_V' is false
TEST(Arnoldi_Ut, is_V_false) {
  int D = 5, d = 2;
  int dim = D * D;
  TMOp H = TMOp(d, D, dim);
  const cytnx_uint64 maxiter = 10000;
  const cytnx_double cvg_crit = 0;
  const std::string which = "LM";
  const cytnx_uint64 k = 3;
  bool is_V = true;
  std::vector<UniTensor> lanczos_isV =
    linalg::Arnoldi(&H, H.T_init, which, maxiter, cvg_crit, k, is_V);
  is_V = false;
  std::vector<UniTensor> lanczos_noV =
    linalg::Arnoldi(&H, H.T_init, which, maxiter, cvg_crit, k, is_V);
  double err = (lanczos_isV[0] - lanczos_noV[0]).Norm().item<double>();
  EXPECT_TRUE(err < 1e-12);
  EXPECT_TRUE(lanczos_noV.size() == 1);
  EXPECT_TRUE(lanczos_isV.size() == (k + 1));
}

// 1-11 Block UniTensor
TEST(Arnoldi_Gnd, Arnoldi_BK_test) {
  Bond lan_I_v = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
  Bond lan_J_v = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {1, 1, 1});
  UniTensor UT_init = UniTensor({lan_I_v, lan_J_v});
  double low = -1.0, high = 1.0;
  int seed = 0;
  UT_init.uniform_(low, high, seed);
  UT_init.relabel_({"b", "c"});

  MyOp2 H = MyOp2(27);

  std::vector<UniTensor> exact_eigs = linalg::Eig(H.H);
  double lambda = -DBL_MAX;
  for (auto& block : exact_eigs[0].get_blocks_()) {
    lambda = std::max(lambda, linalg::Max(block).item<double>());
  }
  std::vector<UniTensor> eigs = linalg::Arnoldi(&H, UT_init, "LM");
  cytnx_double ev = (cytnx_double)eigs[0].get_block_()(0).item().real();
  // std::cout << ev << ' ' << lambda << std::endl;
  // std::cout << "err val=" <<std::abs(ev - lambda) << std::endl;
  EXPECT_TRUE(std::abs(ev - lambda) < 1e-12);
  auto err = (H.matvec(eigs[1]) - ev * eigs[1]).Norm().item();
  // std::cout << "err" << err << std::endl;
  EXPECT_TRUE(err < 1e-12);
}

// error test
class ErrorTestClass {
 public:
  std::string which = "LM";
  cytnx_uint64 k = 1;
  cytnx_uint64 maxiter = 10;
  cytnx_int32 d = 2, D = 5;
  cytnx_double cvg_crit = 0;
  bool is_V = true;
  cytnx_int32 ncv = 0;
  ErrorTestClass(){};
  void ExcuteErrorTest();
  cytnx_uint64 dim = D * D;
  // set
  void Set_dim(const int _dim) {
    dim = _dim;
    H = TMOp(d, D, dim);
  }
  void Set_mat_type(const int _mat_type) {
    mat_type = _mat_type;
    H = TMOp(d, D, dim, mat_type);
  }

 private:
  int mat_type = Type.ComplexDouble;
  TMOp H = TMOp(d, D, dim, mat_type);
};
void ErrorTestClass::ExcuteErrorTest() {
  try {
    auto arnoldi_eigs = linalg::Arnoldi(&H, H.T_init, which, maxiter, cvg_crit, k, is_V, ncv);
    FAIL();
  } catch (const std::exception& ex) {
    auto err_msg = ex.what();
    std::cerr << err_msg << std::endl;
    SUCCEED();
  }
}

// 2-1, test for wrong input 'which'
TEST(Arnoldi_Ut, err_which) {
  ErrorTestClass err_task1;
  err_task1.which = "ML";
  err_task1.ExcuteErrorTest();
}

// 2-2, test SM is not support for UniTensor
TEST(Arnoldi_Ut, err_which_SM) {
  ErrorTestClass err_task;
  err_task.which = "SM";
  err_task.ExcuteErrorTest();
}

// 2-3, test for wrong input LinOp dtype
TEST(Arnoldi_Ut, err_mat_type) {
  ErrorTestClass err_task;
  err_task.Set_mat_type(Type.Int64);
  err_task.ExcuteErrorTest();
}

// 2-4, test for 'k' = 0
TEST(Arnoldi_Ut, err_zero_k) {
  ErrorTestClass err_task;
  err_task.k = 0;
  err_task.ExcuteErrorTest();
}

// 2-5, test for 'k' > 'max_iter'
TEST(Arnoldi_Ut, err_k_too_large) {
  ErrorTestClass err_task;
  err_task.k = err_task.dim + 1;
  err_task.ExcuteErrorTest();
}

// 2-6, test cvg_crit <= 0
TEST(Arnoldi_Ut, err_crit_negative) {
  ErrorTestClass err_task;
  err_task.cvg_crit = -0.001;
  err_task.ExcuteErrorTest();
}

// 2-7, test nx not match
TEST(Arnoldi_Ut, nx_not_match) {
  ErrorTestClass err_task;
  err_task.Set_dim(5);
  err_task.ExcuteErrorTest();
}

// 2-8, test ncv is out of allowd range
TEST(Arnoldi_Ut, err_ncv_out_of_range) {
  ErrorTestClass err_task;
  err_task.ncv = err_task.k + 1;
  err_task.ExcuteErrorTest();
  auto dim = err_task.D * err_task.D * err_task.d;
  err_task.ncv = dim + 1;
  err_task.ExcuteErrorTest();
}
