#ifndef _H_linalg_test
#define _H_linalg_test

#include "cytnx.hpp"
#include <gtest/gtest.h>

using namespace cytnx;
using namespace std;

class linalg_Test : public ::testing::Test {
 public:
  // ==================== general ===================
  Tensor arange3x3d = arange(0, 9, 1, Type.Double).reshape(3, 3);
  Tensor ones3x3d = ones(9, Type.Double).reshape(3, 3);
  Tensor eye3x3d = eye(3, Type.Double);
  Tensor zeros3x3d = zeros(9, Type.Double).reshape(3, 3);

  Tensor arange3x3cd = arange(0, 9, 1, Type.ComplexDouble).reshape(3, 3) +
                       cytnx_complex128(0, 1) * arange(0, 9, 1, Type.ComplexDouble).reshape(3, 3);
  Tensor ones3x3cd = ones(9, Type.ComplexDouble).reshape(3, 3);
  Tensor eye3x3cd = eye(3, Type.ComplexDouble);
  Tensor zeros3x3cd = zeros(9, Type.ComplexDouble).reshape(3, 3);

  Tensor invertable3x3cd = arange(1, 10, 1, Type.ComplexDouble).reshape(3, 3);

  UniTensor arange3x3cd_ut = UniTensor(arange3x3cd, false, -1);
  UniTensor ones3x3cd_ut = UniTensor(ones3x3cd, false, -1);
  UniTensor invertable3x3cd_ut = UniTensor(invertable3x3cd, false, -1);

  std::string data_dir = "../../tests/test_data_base/linalg/";
  // ==================== svd_truncate ===================
  Bond svd_I = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
  Bond svd_J =
    Bond(BD_IN, {Qs(5), Qs(3), Qs(1), Qs(-1), Qs(-3), Qs(-5), Qs(-7)}, {6, 22, 57, 68, 38, 8, 1});
  Bond svd_K =
    Bond(BD_OUT, {Qs(5), Qs(3), Qs(1), Qs(-1), Qs(-3), Qs(-5), Qs(-7)}, {6, 22, 57, 68, 38, 8, 1});
  Bond svd_L = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
  UniTensor svd_T = UniTensor({svd_I, svd_J, svd_K, svd_L}, {"a", "b", "c", "d"}, 1, Type.Double,
                              Device.cpu, false);

  UniTensor svd_T_dense =
    UniTensor(arange(0, 11 * 13, 1).reshape(11, 13)).astype(Type.ComplexDouble).to(Device.cpu);
  Tensor svd_Sans;
  //==================== Lanczos_Gnd_Ut ===================
  Tensor A = Tensor::Load(data_dir + "Lanczos_Gnd/lan_block_A.cytn");
  Tensor B = Tensor::Load(data_dir + "Lanczos_Gnd/lan_block_B.cytn");
  Tensor C = Tensor::Load(data_dir + "Lanczos_Gnd/lan_block_C.cytn");
  Bond lan_I = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
  Bond lan_J = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
  UniTensor H = UniTensor({lan_I, lan_J});

  //==================== QR ===================
  Tensor Qr_Qans = Tensor::Load(data_dir + "Qr/qr_Qans.cytn");
  Tensor Qr_Rans = Tensor::Load(data_dir + "Qr/qr_Rans.cytn");

  //==================== ExpH ===================
  Tensor expH_ans = Tensor::Load(data_dir + "expH/expH_ans.cytn");

  //==================== Pow ===================
  Tensor Pow_ans = Tensor::Load(data_dir + "Pow/Pow_ans.cytn");

  //==================== Mod ===================
  Tensor Mod_ans = Tensor::Load(data_dir + "Mod/Mod_ans.cytn");
  Tensor ModUtUt_ans = Tensor::Load(data_dir + "Mod/ModUtUt_ans.cytn");

 protected:
  void SetUp() override {
    //================ svd truncate =======================
    svd_T = svd_T.Load(data_dir + "Svd_truncate/Svd_truncate1.cytnx");
    svd_T.permute_({1, 0, 3, 2});
    svd_T.contiguous_();
    svd_T.set_rowrank_(2);
    svd_Sans = Tensor::Load(data_dir + "Svd_truncate/S_truncate1.cytn");
    svd_Sans = algo::Sort(svd_Sans);
    //==================== Lanczos_Gnd_Ut ===================
    H.put_block(A, 0);
    H.put_block(B, 1);
    H.put_block(C, 2);
    H.set_labels({"a", "b"});

    invertable3x3cd.at({0, 0}) = 2;  // just to make it invertable.
    invertable3x3cd_ut.at({0, 0}) = 2;  // just to make it invertable.
  }
  void TearDown() override {}
};

#endif
