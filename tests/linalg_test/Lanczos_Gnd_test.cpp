#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "cytnx.hpp"

using namespace cytnx;
using namespace testing;

namespace {
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
      EffH = CreateEffH();
    }
    UniTensor UT_init;
    UniTensor EffH;

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
      return net_.Launch();
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
    UniTensor CreateEffH() {
      Network net;
      net.FromString({"L:'vil', 'Lm', 'vol'", "O:'Lm', 'pi', 'Rm', 'po'", "R:'vir', 'Rm', 'vor'",
                      "TOUT:'vil', 'pi', 'vir'; 'vol', 'po', 'vor'"});
      net.PutUniTensors({"L", "O", "R"}, {L_, O_, R_});
      return net.Launch();
    }
  };

  class MyOp : public LinOp {
   public:
    MyOp() : LinOp("mv", 27) {}

    UniTensor matvec(const UniTensor& v) override {
      Tensor tA = arange(27 * 27).reshape(27, 27);
      UniTensor A = UniTensor(tA);
      // A = A + A.clone().permute({1, 0},-1,false);
      A = A + A.Transpose();
      return UniTensor(linalg::Dot(A.get_block_(), v.get_block_()));
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
      H.set_labels({"a", "b"});
      // H.print_diagram();
      // H.print_blocks();
    }
    UniTensor matvec(const UniTensor& psi) override {
      auto out = H.contract(psi);
      out.set_labels({"b", "c"});
      // out.print_diagram();
      return out;
    }
  };
  // the function to check the answer
  bool CheckResult(OneSiteOp& H, const std::vector<UniTensor>& lanczos_eigs,
                   const std::string& which, const cytnx_uint64 k);

  void ExcuteTest(const std::string& which, const int& mat_type = Type.Double,
                  const cytnx_uint64& k = 3) {
    int D = 5, d = 2, dw = 3;
    OneSiteOp op = OneSiteOp(d, D, dw, mat_type);
    const cytnx_uint64 maxiter = 1000;
    const cytnx_double cvg_crit = 0;
    std::vector<UniTensor> lanczos_eigs =
      linalg::Lanczos(&op, op.UT_init, which, maxiter, cvg_crit, k);
    bool is_pass = CheckResult(op, lanczos_eigs, which, k);
    EXPECT_TRUE(is_pass);
  }

  // error test
  class ErrorTestClass {
   public:
    std::string which = "LM";
    cytnx_uint64 k = 1;
    cytnx_uint64 maxiter = 10;
    cytnx_int32 d = 2, D = 5, dw = 3;
    cytnx_double cvg_crit = 0;
    bool is_V = true;
    cytnx_int32 ncv = 0;
    ErrorTestClass(){};
    void ExcuteErrorTest();
    // set
    void Set_mat_type(const int _mat_type) {
      mat_type = _mat_type;
      op = OneSiteOp(d, D, dw, mat_type);
    }

   private:
    int mat_type = Type.Double;
    OneSiteOp op = OneSiteOp(d, D, dw, mat_type);
  };
  void ErrorTestClass::ExcuteErrorTest() {
    try {
      auto lanczos_eigs = linalg::Lanczos(&op, op.UT_init, which, maxiter, cvg_crit, k, is_V, ncv);
      FAIL();
    } catch (const std::exception& ex) {
      auto err_msg = ex.what();
      std::cerr << err_msg << std::endl;
      SUCCEED();
    }
  }

  // For given 'which' = 'LM', 'SR', ...etc, sort the given eigenvalues.
  bool cmpNorm(const Scalar& l, const Scalar& r) { return abs(l) < abs(r); }
  bool cmpAlgebra(const Scalar& l, const Scalar& r) { return l < r; }
  std::vector<Scalar> OrderEigvals(const UniTensor& eigvals, const std::string& order_type) {
    char small_or_large = order_type[0];  //'S' or 'L'
    char metric_type = order_type[1];  //'M' or 'A'
    auto eigvals_len = eigvals.shape()[0];
    auto ordered_eigvals = std::vector<Scalar>(eigvals_len, Scalar());
    for (cytnx_uint64 i = 0; i < eigvals_len; ++i)
      ordered_eigvals[i] = eigvals.get_block_().storage().at(i);
    std::function<bool(const Scalar&, const Scalar&)> cmpFncPtr;
    if (metric_type == 'M') {
      cmpFncPtr = cmpNorm;
    } else if (metric_type == 'A') {
      cmpFncPtr = cmpAlgebra;
    } else {
    }  // wrong input
    // sort eigenvalues
    if (small_or_large == 'S') {
      std::stable_sort(ordered_eigvals.begin(), ordered_eigvals.end(), cmpFncPtr);
    } else {  // 'L'
      std::stable_sort(ordered_eigvals.rbegin(), ordered_eigvals.rend(), cmpFncPtr);
    }
    return ordered_eigvals;
  }

  // get resigue |Hv - ev|
  Scalar GetResidue(OneSiteOp& H, const Scalar& eigval, const UniTensor& eigvec) {
    UniTensor resi_vec = H.matvec(eigvec) - eigval * eigvec;
    Scalar resi = resi_vec.Norm().item();
    return resi;
  }

  // compare the lanczos results with full spectrum (calculated by the function Eig.)
  bool CheckResult(OneSiteOp& H, const std::vector<UniTensor>& lanczos_eigs,
                   const std::string& which, const cytnx_uint64 k) {
    // get full spectrum (eigenvalues)
    std::vector<UniTensor> full_eigs = linalg::Eigh(H.EffH);
    UniTensor full_eigvals = full_eigs[0];
    std::vector<Scalar> ordered_eigvals = OrderEigvals(full_eigvals, which);
    auto fst_few_eigvals =
      std::vector<Scalar>(ordered_eigvals.begin(), ordered_eigvals.begin() + k);
    UniTensor lanczos_eigvals = lanczos_eigs[0];

    // check the number of the eigenvalues
    cytnx_uint64 lanczos_eigvals_len = lanczos_eigvals.shape()[0];
    auto dtype = H.dtype();
    const double tolerance = (dtype == Type.ComplexFloat || dtype == Type.Float) ? 1.0e-4 : 1.0e-12;
    if (lanczos_eigvals_len != k) return false;
    for (cytnx_uint64 i = 0; i < k; ++i) {
      auto lanczos_eigval = lanczos_eigvals.get_block_().storage().at(i);
      // if k == 1, lanczos_eigvecs will be a rank-1 tensor
      auto lanczos_eigvec = lanczos_eigs[i + 1];
      auto exact_eigval = fst_few_eigvals[i];
      // check eigen value by comparing with the full spectrum results.
      // avoid, for example, lanczos_eigval = 1 + 3j, exact_eigval = 1 - 3j, which = 'LM'
      auto eigval_err = abs(abs(lanczos_eigval) - abs(exact_eigval)) / abs(exact_eigval);
      // std::cout << "eigval err" << eigval_err << std::endl;
      if (eigval_err >= tolerance) return false;
      // check the is the eigenvector correct
      auto resi_err = GetResidue(H, lanczos_eigval, lanczos_eigvec);
      // std::cout << "resi err" << resi_err << std::endl;
      if (resi_err >= tolerance) return false;
      // check phase
    }
    return true;
  }

}  // namespace

// corrected test
// 1-1, test for 'which' = 'LM'
TEST(Lanczos_Ut, which_LM_test) {
  std::string which = "LM";
  ExcuteTest(which);
}

// 1-2, test for 'which' = 'LA'
TEST(Lanczos_Ut, which_LA_test) {
  std::string which = "LA";
  ExcuteTest(which);
}

// 1-3, test for 'which' = 'SA'
TEST(Lanczos_Ut, which_SA_test) {
  std::string which = "SA";
  ExcuteTest(which);
}

// 1-4, test matrix is all allowed data type
TEST(Lanczos_Ut, all_dtype_test) {
  std::string which = "LM";
  std::vector<int> dtypes = {Type.ComplexDouble, Type.ComplexFloat, Type.Double, Type.Float};
  for (auto dtype : dtypes) {
    ExcuteTest(which, dtype);
  }
}

// 1-7, test eigenalue number k = 1
TEST(Lanczos_Ut, k1_test) {
  std::string which = "LM";
  auto mat_type = Type.Double;
  cytnx_uint64 k = 1;
  ExcuteTest(which, mat_type, k);
}

// 1-8, test eigenalue number k is closed to dimension.
TEST(Lanczos_Ut, k_large) {
  std::string which = "LM";
  auto mat_type = Type.Double;
  cytnx_uint64 k;
  k = 23;  // dim = 25
  ExcuteTest(which, mat_type, k);
}

// 1-10, test 'is_V' is false
TEST(Lanczos_Ut, is_V_false) {
  OneSiteOp op = OneSiteOp();
  const cytnx_uint64 maxiter = 1000;
  const cytnx_double cvg_crit = 0;
  const std::string which = "LM";
  const cytnx_uint64 k = 3;
  bool is_V = true;
  std::vector<UniTensor> lanczos_isV =
    linalg::Lanczos(&op, op.UT_init, which, maxiter, cvg_crit, k, is_V);
  is_V = false;
  std::vector<UniTensor> lanczos_noV =
    linalg::Lanczos(&op, op.UT_init, which, maxiter, cvg_crit, k, is_V);
  double err = (lanczos_isV[0] - lanczos_noV[0]).Norm().item<double>();
  EXPECT_TRUE(err < 1e-12);
  EXPECT_TRUE(lanczos_noV.size() == 1);
  EXPECT_TRUE(lanczos_isV.size() == (k + 1));
}

// 2-1, test for wrong input 'which'
TEST(Lanczos_Ut, err_which) {
  ErrorTestClass err_task1;
  err_task1.which = "ML";
  err_task1.ExcuteErrorTest();
}

// 2-2, test SM is not support for UniTensor
TEST(Lanczos_Ut, err_which_SM) {
  ErrorTestClass err_task;
  err_task.which = "SM";
  err_task.ExcuteErrorTest();
}

// 2-3, test for wrong input LinOp dtype
TEST(Lanczos_Ut, err_mat_type) {
  ErrorTestClass err_task;
  err_task.Set_mat_type(Type.Int64);
  err_task.ExcuteErrorTest();
}

// 2-4, test for 'k' = 0
TEST(Lanczos_Ut, err_zero_k) {
  ErrorTestClass err_task;
  err_task.k = 0;
  err_task.ExcuteErrorTest();
}

// 2-5, test for 'k' > 'max_iter'
TEST(Lanczos_Ut, err_k_too_large) {}

// 2-6, test cvg_crit <= 0
TEST(Lanczos_Ut, err_crit_negative) {
  ErrorTestClass err_task;
  err_task.cvg_crit = -0.001;
  err_task.ExcuteErrorTest();
}

// 2-7, test nx not match
TEST(Lanczos_Ut, nx_not_match) {
  LinOp op = LinOp("mv", 30);
  double low = -1.0, high = 1.0;
  int D = 5, d = 2;
  UniTensor psi = UniTensor::uniform({D, d, D}, low, high);
  try {
    unsigned long long maxiter = 1000;
    double crit = 0;
    unsigned long long k = 3;
    std::string which = "LM";
    auto eigs = linalg::Lanczos(&op, psi, which, maxiter, crit, k);
    FAIL();
  } catch (const std::exception& ex) {
    auto err_msg = ex.what();
    std::cerr << err_msg << std::endl;
    SUCCEED();
  }
}

// 2-8, test ncv is out of allowd range
TEST(Lanczos_Ut, err_ncv_out_of_range) {
  ErrorTestClass err_task;
  err_task.ncv = err_task.k + 1;
  err_task.ExcuteErrorTest();
  auto dim = err_task.D * err_task.D * err_task.d;
  err_task.ncv = dim + 1;
  err_task.ExcuteErrorTest();
}

TEST(Lanczos_Gnd, Lanczos_Gnd_test) {
  // CompareWithScipy

  // cytnx_double evans = -0.6524758424985271;
  cytnx_double evans = -1628.9964650426593;

  // Tensor testtmp = arange(16).reshape(4, 4);
  // std::cout<<testtmp<<std::endl;
  MyOp H = MyOp();
  Tensor tv = arange(27);
  UniTensor v = UniTensor(tv);
  std::vector<UniTensor> eigs =
    linalg::Lanczos(&H, v, "Gnd", 9.999999999999999988e-15, 10000, 1, false, true, 0, false);
  cytnx_double ev = (cytnx_double)eigs[0].get_block_()(0).item().real();
  // std::cout << ev << ' ' << evans << std::endl;
  EXPECT_TRUE(std::fabs(ev - evans) < 1e-5);
  // EXPECT_DOUBLE_EQ(ev, evans);
}

TEST(Lanczos_Gnd, Bk_Lanczos_Gnd_test) {
  // CompareWithScipy
  // cytnx_double evans = -2.31950925;

  Bond lan_I_v = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
  Bond lan_J_v = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {1, 1, 1});
  UniTensor lan_guess = UniTensor({lan_I_v, lan_J_v});

  lan_guess.put_block(random::normal(9, 1, 1).reshape({9, 1}), 0);
  lan_guess.put_block(random::normal(9, 1, 1).reshape({9, 1}), 1);
  lan_guess.put_block(random::normal(9, 1, 1).reshape({9, 1}), 2);
  lan_guess.set_labels({"b", "c"});
  // lan_guess.print_diagram();
  // std::cout << lan_guess.shape() << std::endl;
  //  lan_guess.print_blocks();

  MyOp2 H = MyOp2(27);

  std::vector<UniTensor> exact_eigs = linalg::Eigh(H.H);
  double E0 = DBL_MAX;
  for (auto& block : exact_eigs[0].get_blocks_()) {
    E0 = std::min(E0, linalg::Min(block).item<double>());
  }
  std::vector<UniTensor> eigs =
    linalg::Lanczos(&H, lan_guess, "Gnd", 9.999999999999999988e-15, 10000, 1, true, true, 0, false);
  cytnx_double ev = (cytnx_double)eigs[0].get_block_()(0).item().real();
  // std::cout << ev << ' ' << E0 << std::endl;
  EXPECT_TRUE(std::abs(ev - E0) < 1e-12);
  auto err = (H.matvec(eigs[1]) - ev * eigs[1]).Norm().item();
  EXPECT_TRUE(err < 1e-6);
  // EXPECT_DOUBLE_EQ(ev, evans);
}

TEST(Lanczos_Gnd, Bk_Lanczos_test) {
  Bond lan_I_v = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
  Bond lan_J_v = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {1, 1, 1});
  UniTensor lan_guess = UniTensor({lan_I_v, lan_J_v});

  lan_guess.put_block(random::normal(9, 1, 1).reshape({9, 1}), 0);
  lan_guess.put_block(random::normal(9, 1, 1).reshape({9, 1}), 1);
  lan_guess.put_block(random::normal(9, 1, 1).reshape({9, 1}), 2);
  lan_guess.set_labels({"b", "c"});

  MyOp2 H = MyOp2(27);

  std::vector<UniTensor> exact_eigs = linalg::Eigh(H.H);
  double E0 = DBL_MAX;
  for (auto& block : exact_eigs[0].get_blocks_()) {
    E0 = std::min(E0, linalg::Min(block).item<double>());
  }
  const cytnx_uint64 maxiter = 1000;
  const cytnx_double cvg_crit = 0;
  std::vector<UniTensor> eigs = linalg::Lanczos(&H, lan_guess, "SA", maxiter, cvg_crit);
  cytnx_double ev = (cytnx_double)eigs[0].get_block_()(0).item().real();
  // std::cout << ev << ' ' << E0 << std::endl;
  auto err_val = ev - E0;
  EXPECT_TRUE(std::abs(ev - E0) < 1e-12);
  auto err = (H.matvec(eigs[1]) - ev * eigs[1]).Norm().item();
  EXPECT_TRUE(err < 1e-12);
  // EXPECT_DOUBLE_EQ(ev, evans);
}
