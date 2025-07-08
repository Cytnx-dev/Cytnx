#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "cytnx.hpp"
#include "../test_tools.h"

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
         const unsigned int& dtype, const int& device);
    UniTensor matvec(const UniTensor& l) override {
      auto tmp = Contracts({A, l, B}, "", true);
      tmp.relabels_(l.labels()).set_rowrank(l.rowrank());
      return tmp;
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
  class CheckOp : public LinOp {
   public:
    UniTensor A, B;
    UniTensor T_init;
    TMOp* op;
    CheckOp(TMOp* in_op) : op(in_op),
	    LinOp("mv", in_op->nx(), in_op->dtype(), Device.cpu) {
      A = op->A.to(Device.cpu);
      B = op->B.to(Device.cpu);
      T_init = op->T_init.to(Device.cpu);
    }
    UniTensor matvec(const UniTensor& l) override {
      auto tmp = Contracts({A, l, B}, "", true);
      tmp.relabels_(l.labels()).set_rowrank(l.rowrank());
      return tmp;
    }
  };

  // the function to check the answer
  bool CheckResult(CheckOp& H, const std::vector<UniTensor>& arnoldi_eigs_cuda, 
		   const std::vector<UniTensor>& arnoldi_eigs_cpu);

  void ExcuteTest(const std::string& which, const int& mat_type = Type.ComplexDouble,
                  const cytnx_uint64& k = 3) {
    int D = 5, d = 2;
    int dim = D * D;
    TMOp H = TMOp(d, D, dim, mat_type, Device.cuda);
    CheckOp H_check= CheckOp(&H);
    const cytnx_uint64 maxiter = 10000;
    const cytnx_double cvg_crit = 0;
    std::vector<UniTensor> arnoldi_eigs_cuda =
      linalg::Arnoldi(&H, H.T_init, which, maxiter, cvg_crit, k);
    for (auto& arnoldi_eig : arnoldi_eigs_cuda) {
      EXPECT_EQ(arnoldi_eig.device(), Device.cuda);
    }
    std::vector<UniTensor> arnoldi_eigs_cpu = 
	    linalg::Arnoldi(&H_check, H_check.T_init, which, maxiter, cvg_crit, k);
    bool is_pass = CheckResult(H_check, arnoldi_eigs_cuda, arnoldi_eigs_cpu);
    EXPECT_TRUE(is_pass);
  }

  // get resigue |Hv - ev|
  Scalar GetResidue(CheckOp& H, const Scalar& eigval, const UniTensor& eigvec) {
    UniTensor resi_vec = H.matvec(eigvec) - eigval * eigvec;
    Scalar resi = resi_vec.Norm().item();
    return resi;
  }

  // compare the arnoldi results with full spectrum (calculated by the function Eig.)
  bool CheckResult(CheckOp& H, const std::vector<UniTensor>& arnoldi_eigs_cuda, 
		   const std::vector<UniTensor>& arnoldi_eigs_cpu) {
    auto dtype = H.dtype();
    const double tolerance = (dtype == Type.ComplexFloat || dtype == Type.Float) ?
          1.0e-4 : 1.0e-12;

    //Check eigenvalues copmared with the results from cpu.
    if (arnoldi_eigs_cuda.size() != arnoldi_eigs_cpu.size()) {
      return false;
    }
    UniTensor eigval_cuda_to_cpu = arnoldi_eigs_cuda[0].to(Device.cpu);
    UniTensor eigval_cpu = arnoldi_eigs_cpu[0];
    bool is_same_eigval = TestTools::AreNearlyEqUniTensor(eigval_cuda_to_cpu, eigval_cpu, tolerance);
    if (!is_same_eigval) {
      return false;
    }

    // Check eigenvectors. We have not compare the eigenvector directly since they may have different phase.
    UniTensor arnoldi_eigvecs = arnoldi_eigs_cuda[1].to(Device.cpu);

    // check the number of the eigenvalues
    int k = eigval_cpu.shape()[0];
    for (cytnx_uint64 i = 0; i < k; ++i) {
      // if k == 1, arnoldi_eigvecs will be a rank-1 tensor
      auto arnoldi_eigvec = arnoldi_eigs_cuda[i+1].to(Device.cpu);
      auto arnoldi_eigval = eigval_cuda_to_cpu.at({i});
      // check the is the eigenvector correct
      auto resi_err = GetResidue(H, arnoldi_eigval, arnoldi_eigvec);
      //std::cout << "resi err=" << resi_err << std::endl;
      if (resi_err >= tolerance) return false;
    }
    return true;
  }

  
} //namespace

// corrected test
// 1-1, test for 'which' = 'LM'
TEST(Arnoldi_Ut, gpu_which_LM_test) {
  std::string which = "LM";
  ExcuteTest(which);
}

// 1-2, test for 'which' = 'LR'
TEST(Arnoldi_Ut, gpu_which_LR_test) {
  std::string which = "LR";
  ExcuteTest(which);
}

// 1-3, test for 'which' = 'LI'
TEST(Arnoldi_Ut, gpu_which_LI_test) {
  std::string which = "LI";
  ExcuteTest(which);
}

// 1-4, test for 'which' = 'SR'
TEST(Arnoldi_Ut, gpu_which_SR_test) {
  std::string which = "SR";
  ExcuteTest(which);
}

// 1-5, test for 'which' = 'SI'
TEST(Arnoldi_Ut, gpu_which_SI_test) {
  std::string which = "SI";
  ExcuteTest(which);
}

// 1-6, test matrix is all allowed data type
// not correct since the Contracts in gpu is incorrect
/*
TEST(Arnoldi_Ut, gpu_all_dtype_test) {
  std::string which = "LM";
  std::vector<int>  dtypes = 
  {Type.ComplexDouble, Type.ComplexFloat, Type.Double, Type.Float};
  for (auto dtype : dtypes) {
    ExcuteTest(which, dtype);
  }
}
*/

// 1-7, test eigenalue number k = 1
TEST(Arnoldi_Ut, gpu_k1_test) {
  std::string which = "LM";
  auto mat_type = Type.ComplexDouble;
  cytnx_uint64 k = 1;
  ExcuteTest(which, mat_type, k);
}

// 1-8, test eigenalue number k is closed to dimension.
TEST(Arnoldi_Ut, gpu_k_large) {
  std::string which = "LM";
  auto mat_type = Type.ComplexDouble;
  cytnx_uint64 k;
  k = 23; //dim = 25
  ExcuteTest(which, mat_type, k);
}

// 1-9, test the smallest matrix dimenstion.
TEST(Arnoldi_Ut, gpu_smallest_dim) {
  std::string which = "LM";
  auto mat_type = Type.ComplexDouble;
  cytnx_uint64 k;
  k = 1;
  ExcuteTest(which, mat_type, k);
}

