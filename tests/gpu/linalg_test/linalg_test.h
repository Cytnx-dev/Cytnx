#ifndef _H_linalg_test
#define _H_linalg_test

#include "cytnx.hpp"
#include <gtest/gtest.h>

using namespace cytnx;
using namespace std;

class linalg_Test : public ::testing::Test {
 public:
  // ==================== general ===================
  Tensor arange3x3d = arange(0, 9, 1, Type.Double).reshape(3, 3).to(cytnx::Device.cuda);
  Tensor ones3x3d = ones(9, Type.Double).reshape(3, 3).to(cytnx::Device.cuda);
  Tensor eye3x3d = eye(3, Type.Double).to(cytnx::Device.cuda);
  Tensor zeros3x3d = zeros(9, Type.Double).reshape(3, 3).to(cytnx::Device.cuda);

  Tensor arange3x3cd = arange(0, 9, 1, Type.ComplexDouble).reshape(3, 3).to(cytnx::Device.cuda) +
                       cytnx_complex128(0, 1) *
                         arange(0, 9, 1, Type.ComplexDouble).reshape(3, 3).to(cytnx::Device.cuda);
  Tensor ones3x3cd = ones(9, Type.ComplexDouble).reshape(3, 3).to(cytnx::Device.cuda);
  Tensor eye3x3cd = eye(3, Type.ComplexDouble).to(cytnx::Device.cuda);
  Tensor zeros3x3cd = zeros(9, Type.ComplexDouble).reshape(3, 3).to(cytnx::Device.cuda);

  std::string data_dir = "../../../tests/test_data_base/linalg/";
  // ==================== svd_truncate ===================
  Bond svd_I = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
  Bond svd_J =
    Bond(BD_IN, {Qs(5), Qs(3), Qs(1), Qs(-1), Qs(-3), Qs(-5), Qs(-7)}, {6, 22, 57, 68, 38, 8, 1});
  Bond svd_K =
    Bond(BD_OUT, {Qs(5), Qs(3), Qs(1), Qs(-1), Qs(-3), Qs(-5), Qs(-7)}, {6, 22, 57, 68, 38, 8, 1});
  Bond svd_L = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
  UniTensor svd_T = UniTensor({svd_I, svd_J, svd_K, svd_L}, {"a", "b", "c", "d"}, 1, Type.Double,
                              Device.cuda, false)
                      .to(cytnx::Device.cuda);

  UniTensor svd_T_dense =
    UniTensor(arange(0, 11 * 13, 1).reshape(11, 13)).astype(Type.ComplexDouble).to(Device.cuda);

  Tensor svd_Sans;
  //==================== Lanczos_Gnd_Ut ===================
  Tensor A = Tensor::Load(data_dir + "Lanczos_Gnd/lan_block_A.cytn").to(cytnx::Device.cuda);
  Tensor B = Tensor::Load(data_dir + "Lanczos_Gnd/lan_block_B.cytn").to(cytnx::Device.cuda);
  Tensor C = Tensor::Load(data_dir + "Lanczos_Gnd/lan_block_C.cytn").to(cytnx::Device.cuda);
  Bond lan_I = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
  Bond lan_J = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
  UniTensor H = UniTensor({lan_I, lan_J}).to(cytnx::Device.cuda);

  //==================== QR ===================
  Tensor Qr_Qans = Tensor::Load(data_dir + "Qr/qr_Qans.cytn").to(cytnx::Device.cuda);
  Tensor Qr_Rans = Tensor::Load(data_dir + "Qr/qr_Rans.cytn").to(cytnx::Device.cuda);

  //==================== ExpH ===================
  Tensor expH_ans = Tensor::Load(data_dir + "expH/expH_ans.cytn").to(cytnx::Device.cuda);

  //==================== Pow ===================
  Tensor Pow_ans = Tensor::Load(data_dir + "Pow/Pow_ans.cytn").to(cytnx::Device.cuda);

  //==================== Mod ===================
  Tensor Mod_ans = Tensor::Load(data_dir + "Mod/Mod_ans.cytn").to(cytnx::Device.cuda);
  Tensor ModUtUt_ans = Tensor::Load(data_dir + "Mod/ModUtUt_ans.cytn").to(cytnx::Device.cuda);

 protected:
  void SetUp() override {
    //================ svd truncate =======================

    svd_T = svd_T.Load(data_dir + "Svd_truncate/Svd_truncate1.cytnx").to(cytnx::Device.cuda);
    svd_T.permute_({1, 0, 3, 2});
    svd_T.contiguous_();
    svd_T.set_rowrank_(2);
    svd_Sans = Tensor::Load(data_dir + "Svd_truncate/S_truncate1.cytn").to(cytnx::Device.cuda);
    svd_Sans = algo::Sort(svd_Sans);
    //==================== Lanczos_Gnd_Ut ===================
    H.put_block(A, 0);
    H.put_block(B, 1);
    H.put_block(C, 2);
    H.set_labels({"a", "b"});
  }
  void TearDown() override {}
};

#endif
