#include "linalg_test.h"

TEST_F(linalg_Test, BkUt_Svd_truncate1) {
  std::vector<UniTensor> res = linalg::Svd_truncate(svd_T, 200, 0, true);
  std::vector<double> vnm_S;
  for (size_t i = 0; i < res[0].shape()[0]; i++)
    vnm_S.push_back((double)(res[0].at({i, i}).real()));
  std::sort(vnm_S.begin(), vnm_S.end());
  for (size_t i = 0; i < vnm_S.size(); i++)
    EXPECT_TRUE(abs(vnm_S[i] - (double)(svd_Sans.at({0, i}).real())) < 1e-5);
  auto con_T1 = Contract(Contract(res[2], res[0]), res[1]);
  auto con_T2 = Contract(Contract(res[1], res[0]), res[2]);
}

TEST_F(linalg_Test, BkUt_Svd_truncate2) {
  std::vector<UniTensor> res = linalg::Svd_truncate(svd_T, 200, 1e-1, true);
  std::vector<double> vnm_S;
  for (size_t i = 0; i < res[0].shape()[0]; i++)
    vnm_S.push_back((double)(res[0].at({i, i}).real()));
  std::sort(vnm_S.begin(), vnm_S.end());
  for (size_t i = 0; i < vnm_S.size(); i++) EXPECT_TRUE(vnm_S[i] > 1e-1);
  auto con_T1 = Contract(Contract(res[2], res[0]), res[1]);
  auto con_T2 = Contract(Contract(res[1], res[0]), res[2]);
}

TEST_F(linalg_Test, BkUt_Svd_truncate3) {
  Bond I = Bond(BD_IN, {Qs(-5), Qs(-3), Qs(-1), Qs(1), Qs(3), Qs(5)}, {1, 4, 10, 9, 5, 1});
  Bond J = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
  Bond K = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
  Bond L = Bond(BD_OUT, {Qs(-5), Qs(-3), Qs(-1), Qs(1), Qs(3), Qs(5)}, {1, 4, 10, 9, 5, 1});
  UniTensor cyT = UniTensor({I, J, K, L}, {"a", "b", "c", "d"}, 2, Type.Double, Device.cpu, false);
  auto cyT2 = UniTensor::Load(data_dir + "Svd_truncate/Svd_truncate2.cytnx");
  std::vector<UniTensor> res = linalg::Svd_truncate(cyT, 30, 0, true);
  auto con_T1 = Contract(Contract(res[2], res[0]), res[1]);
  auto con_T2 = Contract(Contract(res[1], res[0]), res[2]);
}

TEST_F(linalg_Test, BkUt_Svd_truncate4) {
  Bond I = Bond(BD_IN, {Qs(-4), Qs(-2), Qs(0), Qs(2), Qs(4)}, {2, 7, 10, 8, 3});
  Bond J = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
  Bond K = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
  Bond L = Bond(BD_OUT, {Qs(-4), Qs(-2), Qs(0), Qs(2), Qs(4), Qs(6)}, {1, 5, 10, 9, 4, 1});
  UniTensor cyT = UniTensor({I, J, K, L}, {"a", "b", "c", "d"}, 2, Type.Double, Device.cpu, false);
  cyT = UniTensor::Load(data_dir + "Svd_truncate/Svd_truncate3.cytnx");
  std::vector<UniTensor> res = linalg::Svd_truncate(cyT, 30, 0, true);
  auto con_T1 = Contract(Contract(res[2], res[0]), res[1]);
  auto con_T2 = Contract(Contract(res[1], res[0]), res[2]);
}

TEST_F(linalg_Test, BkUt_Qr1) {
  auto res = linalg::Qr(H);
  auto Q = res[0];
  auto R = res[1];
  for (size_t i = 0; i < 27; i++)
    for (size_t j = 0; j < 27; j++) {
      if (R.elem_exists({i, j})) {
        EXPECT_TRUE(abs((double)(R.at({i, j}).real()) - (double)(Qr_Rans.at({i, j}).real())) <
                    1E-12);
        // EXPECT_EQ((double)(R.at({i,j}).real()),(double)(Qr_Rans.at({i,j}).real()));
      }
      if (Q.elem_exists({i, j})) {
        EXPECT_TRUE(abs((double)(Q.at({i, j}).real()) - (double)(Qr_Qans.at({i, j}).real())) <
                    1E-12);
        // EXPECT_EQ((double)(Q.at({i,j}).real()),(double)(Qr_Qans.at({i,j}).real()));
      }
    }
}

TEST_F(linalg_Test, BkUt_expH) {
  auto res = linalg::ExpH(H);
  for (size_t i = 0; i < 27; i++)
    for (size_t j = 0; j < 27; j++) {
      if (res.elem_exists({i, j})) {
        EXPECT_TRUE(abs((double)(res.at({i, j}).real()) - (double)(expH_ans.at({i, j}).real())) <
                    1E-8);
        // EXPECT_EQ((double)(R.at({i,j}).real()),(double)(Qr_Rans.at({i,j}).real()));
      }
    }
}

TEST_F(linalg_Test, BkUt_expM) {
  auto res = linalg::ExpM(H);
  for (size_t i = 0; i < 27; i++)
    for (size_t j = 0; j < 27; j++) {
      if (res.elem_exists({i, j})) {
        EXPECT_TRUE(abs((double)(res.at({i, j}).real()) - (double)(expH_ans.at({i, j}).real())) <
                    1E-8);
        // EXPECT_EQ((double)(R.at({i,j}).real()),(double)(Qr_Rans.at({i,j}).real()));
      }
    }
}

TEST_F(linalg_Test, DenseUt_Pow) {
  UniTensor Ht = UniTensor(A);
  auto res = linalg::Pow(Ht, 3);
  for (size_t i = 0; i < 9; i++)
    for (size_t j = 0; j < 9; j++) {
      // if(res.elem_exists({i,j})){
      EXPECT_TRUE(abs((double)(res.at({i, j}).real()) - (double)(Pow_ans.at({i, j}).real())) <
                  1E-8);
      // EXPECT_EQ((double)(R.at({i,j}).real()),(double)(Qr_Rans.at({i,j}).real()));
      //}
    }
}

TEST_F(linalg_Test, DenseUt_Pow_) {
  UniTensor Ht = UniTensor(A);
  linalg::Pow_(Ht, 3);
  for (size_t i = 0; i < 9; i++)
    for (size_t j = 0; j < 9; j++) {
      // if(Ht.elem_exists({i,j})){
      EXPECT_TRUE(abs((double)(Ht.at({i, j}).real()) - (double)(Pow_ans.at({i, j}).real())) < 1E-8);
      // EXPECT_EQ((double)(R.at({i,j}).real()),(double)(Qr_Rans.at({i,j}).real()));
      //}
    }
}

TEST_F(linalg_Test, DenseUt_Mod) {
  UniTensor At = UniTensor(A);
  auto res = linalg::Mod(100 * At, 3);
  for (size_t i = 0; i < 9; i++)
    for (size_t j = 0; j < 9; j++) {
      // if(Ht.elem_exists({i,j})){
      EXPECT_TRUE(abs((double)(res.at({i, j}).real()) - (double)(Mod_ans.at({i, j}).real())) <
                  1E-8);
      // EXPECT_EQ((double)(R.at({i,j}).real()),(double)(Qr_Rans.at({i,j}).real()));
      //}
    }
}

TEST_F(linalg_Test, Tensor_Gemm) {
  Tensor res_d = linalg::Gemm(0.5, arange3x3d, eye3x3d);
  Tensor ans_d = arange3x3d * 0.5;

  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ(res_d(i, j).item(), ans_d(i, j).item());
    }

  Tensor res_cd = linalg::Gemm(0.5, arange3x3cd, eye3x3cd);
  Tensor ans_cd = arange3x3cd * 0.5;

  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ(res_cd(i, j).item().real(), ans_cd(i, j).item().real());
      EXPECT_EQ(res_cd(i, j).item().imag(), ans_cd(i, j).item().imag());
    }
}

TEST_F(linalg_Test, Tensor_Gemm_) {
  Tensor C_d = arange3x3d.clone();
  linalg::Gemm_(1, arange3x3d, eye3x3d, 0.5, C_d);
  Tensor ans_d = arange3x3d * 1.5;
  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ(C_d(i, j).item(), ans_d(i, j).item());
    }

  Tensor C_cd = arange3x3cd.clone();
  linalg::Gemm_(1, arange3x3cd, eye3x3cd, 0.5, C_cd);
  Tensor ans_cd = arange3x3cd * 1.5;

  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ(C_cd(i, j).item().real(), ans_cd(i, j).item().real());
      EXPECT_EQ(C_cd(i, j).item().imag(), ans_cd(i, j).item().imag());
    }
}

// TEST_F(linalg_Test, DenseUt_Mod_UtUt){
//     UniTensor At = UniTensor(A);
//     UniTensor Bt = UniTensor(B);
//     auto res = linalg::Mod(100*At, Bt);
//     for(size_t i = 0;i<9;i++)
//       for(size_t j = 0; j<9;j++){
//           //if(Ht.elem_exists({i,j})){
//             EXPECT_TRUE(abs((double)(res.at({i,j}).real())-(double)(ModUtUt_ans.at({i,j}).real()))
//             < 1E-8);
//             //EXPECT_EQ((double)(R.at({i,j}).real()),(double)(Qr_Rans.at({i,j}).real()));
//           //}
//       }
// }
