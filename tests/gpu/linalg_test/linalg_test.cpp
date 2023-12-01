#include "linalg_test.h"

TEST_F(linalg_Test, gpu_BkUt_Svd_truncate1) {
  std::vector<UniTensor> res = linalg::Svd_truncate(svd_T, 200, 0, true);
  std::vector<double> vnm_S;
  for (size_t i = 0; i < res[0].shape()[0]; i++)
    vnm_S.push_back((double)(res[0].at({i, i}).real()));
  std::sort(vnm_S.begin(), vnm_S.end());
  for (size_t i = 0; i < vnm_S.size(); i++) {
    EXPECT_TRUE(abs(vnm_S[i] - (double)(svd_Sans.at({0, i}).real())) < 1e-4);
    // EXPECT_DOUBLE_EQ(vnm_S[i], (double)(svd_Sans.at({0, i}).real()));
  }
  auto con_T1 = Contract(Contract(res[2], res[0]), res[1]);
  auto con_T2 = Contract(Contract(res[1], res[0]), res[2]);
}

TEST_F(linalg_Test, gpu_BkUt_Svd_truncate2) {
  std::vector<UniTensor> res = linalg::Svd_truncate(svd_T, 200, 1e-1, true);
  std::vector<double> vnm_S;
  for (size_t i = 0; i < res[0].shape()[0]; i++)
    vnm_S.push_back((double)(res[0].at({i, i}).real()));
  std::sort(vnm_S.begin(), vnm_S.end());
  for (size_t i = 0; i < vnm_S.size(); i++) EXPECT_TRUE(vnm_S[i] > 1e-1);
  auto con_T1 = Contract(Contract(res[2], res[0]), res[1]);
  auto con_T2 = Contract(Contract(res[1], res[0]), res[2]);
}

TEST_F(linalg_Test, gpu_BkUt_Svd_truncate3) {
  std::vector<UniTensor> res = linalg::Svd_truncate(svd_T, 200, 0, true);
  UniTensor densvd_T = UniTensor(zeros(svd_T.shape(), svd_T.dtype(), svd_T.device()));
  std::vector<UniTensor> denres = linalg::Svd_truncate(densvd_T.convert_from(svd_T), 200, 0, true);

  std::vector<double> vnm_S, denvnm_S;
  for (size_t i = 0; i < res[0].shape()[0]; i++)
    vnm_S.push_back((double)(res[0].at({i, i}).real()));
  for (size_t i = 0; i < denres[0].shape()[0]; i++)
    denvnm_S.push_back((double)(denres[0].at({i, i}).real()));
  std::sort(vnm_S.begin(), vnm_S.end());
  std::sort(denvnm_S.begin(), denvnm_S.end());
  for (size_t i = 0; i < vnm_S.size(); i++) {
    EXPECT_DOUBLE_EQ(vnm_S[i], denvnm_S[i]);
  }
  // auto con_T1 = Contract(Contract(res[2], res[0]), res[1]);
  // auto con_T2 = Contract(Contract(res[1], res[0]), res[2]);
}

// TEST_F(linalg_Test, gpu_BkUt_Svd_truncate3) {
//   Bond I = Bond(BD_IN, {Qs(-5), Qs(-3), Qs(-1), Qs(1), Qs(3), Qs(5)}, {1, 4, 10, 9, 5, 1});
//   Bond J = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
//   Bond K = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
//   Bond L = Bond(BD_OUT, {Qs(-5), Qs(-3), Qs(-1), Qs(1), Qs(3), Qs(5)}, {1, 4, 10, 9, 5, 1});
//   UniTensor cyT = UniTensor({I, J, K, L}, {"a", "b", "c", "d"}, 2, Type.Double, Device.cuda,
//   false)
//                     .to(cytnx::Device.cuda);
//   auto cyT2 = UniTensor::Load(data_dir +
//   "Svd_truncate/Svd_truncate2.cytnx").to(cytnx::Device.cuda); std::vector<UniTensor> res =
//   linalg::Svd_truncate(cyT, 30, 0, true); auto con_T1 = Contract(Contract(res[2], res[0]),
//   res[1]); auto con_T2 = Contract(Contract(res[1], res[0]), res[2]);
// }

// TEST_F(linalg_Test, gpu_BkUt_Svd_truncate4) {
//   Bond I = Bond(BD_IN, {Qs(-4), Qs(-2), Qs(0), Qs(2), Qs(4)}, {2, 7, 10, 8, 3});
//   Bond J = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
//   Bond K = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
//   Bond L = Bond(BD_OUT, {Qs(-4), Qs(-2), Qs(0), Qs(2), Qs(4), Qs(6)}, {1, 5, 10, 9, 4, 1});
//   UniTensor cyT = UniTensor({I, J, K, L}, {"a", "b", "c", "d"}, 2, Type.Double, Device.cuda,
//   false)
//                     .to(cytnx::Device.cuda);
//   cyT = UniTensor::Load(data_dir + "Svd_truncate/Svd_truncate3.cytnx").to(cytnx::Device.cuda);
//   std::vector<UniTensor> res = linalg::Svd_truncate(cyT, 30, 0, true);
//   auto con_T1 = Contract(Contract(res[2], res[0]), res[1]);
//   auto con_T2 = Contract(Contract(res[1], res[0]), res[2]);
// }

TEST_F(linalg_Test, gpu_BkUt_Qr1) {
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

TEST_F(linalg_Test, gpu_BkUt_expH) {
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

TEST_F(linalg_Test, gpu_BkUt_expM) {
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

TEST_F(linalg_Test, gpu_DenseUt_Gesvd_truncate) {
  std::vector<UniTensor> full = linalg::Gesvd_truncate(svd_T_dense, 999, 0, true, true, 999);
  EXPECT_EQ(full[0].shape()[0], 11);

  EXPECT_EQ(full[1].shape()[0], 11);
  EXPECT_EQ(full[1].shape()[1], 11);

  EXPECT_EQ(full[2].shape()[0], 11);
  EXPECT_EQ(full[2].shape()[1], 13);

  std::vector<UniTensor> truc1 = linalg::Gesvd_truncate(svd_T_dense, 5, 0, true, true, 999);

  EXPECT_EQ(truc1[0].shape()[0], 5);

  EXPECT_EQ(truc1[1].shape()[0], 11);
  EXPECT_EQ(truc1[1].shape()[1], 5);

  EXPECT_EQ(truc1[2].shape()[0], 5);
  EXPECT_EQ(truc1[2].shape()[1], 13);

  EXPECT_EQ(truc1[3].shape()[0], 6);

  for (size_t i = 0; i < 5; i++) {
    EXPECT_EQ(full[0].at({i}), truc1[0].at({i}));
  }
  for (size_t i = 0; i < 6; i++) {
    EXPECT_EQ(full[0].at({i + 5}), truc1[3].at({i}));
  }

  for (size_t i = 0; i < 11; i++) {
    for (size_t j = 0; j < 5; j++) {
      EXPECT_EQ(full[1].at({i, j}), truc1[1].at({i, j}));
    }
  }
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 13; j++) {
      EXPECT_EQ(full[2].at({i, j}), truc1[2].at({i, j}));
    }
  }

  std::vector<UniTensor> truc2 = linalg::Gesvd_truncate(svd_T_dense, 5, 1e-12, true, true, 999);

  EXPECT_EQ(truc2[0].shape()[0], 2);

  EXPECT_EQ(truc2[1].shape()[0], 11);
  EXPECT_EQ(truc2[1].shape()[1], 2);

  EXPECT_EQ(truc2[2].shape()[0], 2);
  EXPECT_EQ(truc2[2].shape()[1], 13);

  EXPECT_EQ(truc2[3].shape()[0], 9);

  for (size_t i = 0; i < 2; i++) {
    EXPECT_EQ(full[0].at({i}), truc2[0].at({i}));
  }
  for (size_t i = 0; i < 9; i++) {
    EXPECT_EQ(full[0].at({i + 2}), truc2[3].at({i}));
  }

  for (size_t i = 0; i < 11; i++) {
    for (size_t j = 0; j < 2; j++) {
      EXPECT_EQ(full[1].at({i, j}), truc2[1].at({i, j}));
    }
  }
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 13; j++) {
      EXPECT_EQ(full[2].at({i, j}), truc2[2].at({i, j}));
    }
  }
}

TEST_F(linalg_Test, gpu_DenseUt_Svd_truncate) {
  std::vector<UniTensor> full = linalg::Svd_truncate(svd_T_dense, 999, 0, true, 999);
  EXPECT_EQ(full[0].shape()[0], 11);

  EXPECT_EQ(full[1].shape()[0], 11);
  EXPECT_EQ(full[1].shape()[1], 11);

  EXPECT_EQ(full[2].shape()[0], 11);
  EXPECT_EQ(full[2].shape()[1], 13);

  std::vector<UniTensor> truc1 = linalg::Svd_truncate(svd_T_dense, 5, 0, true, 999);

  EXPECT_EQ(truc1[0].shape()[0], 5);

  EXPECT_EQ(truc1[1].shape()[0], 11);
  EXPECT_EQ(truc1[1].shape()[1], 5);

  EXPECT_EQ(truc1[2].shape()[0], 5);
  EXPECT_EQ(truc1[2].shape()[1], 13);

  EXPECT_EQ(truc1[3].shape()[0], 6);

  for (size_t i = 0; i < 5; i++) {
    EXPECT_EQ(full[0].at({i}), truc1[0].at({i}));
  }
  for (size_t i = 0; i < 6; i++) {
    EXPECT_EQ(full[0].at({i + 5}), truc1[3].at({i}));
  }

  for (size_t i = 0; i < 11; i++) {
    for (size_t j = 0; j < 5; j++) {
      EXPECT_EQ(full[1].at({i, j}), truc1[1].at({i, j}));
    }
  }
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 13; j++) {
      EXPECT_EQ(full[2].at({i, j}), truc1[2].at({i, j}));
    }
  }

  std::vector<UniTensor> truc2 = linalg::Svd_truncate(svd_T_dense, 5, 1e-9, true, 999);

  EXPECT_EQ(truc2[0].shape()[0], 2);

  EXPECT_EQ(truc2[1].shape()[0], 11);
  EXPECT_EQ(truc2[1].shape()[1], 2);

  EXPECT_EQ(truc2[2].shape()[0], 2);
  EXPECT_EQ(truc2[2].shape()[1], 13);

  EXPECT_EQ(truc2[3].shape()[0], 9);

  for (size_t i = 0; i < 2; i++) {
    EXPECT_EQ(full[0].at({i}), truc2[0].at({i}));
  }
  for (size_t i = 0; i < 9; i++) {
    EXPECT_EQ(full[0].at({i + 2}), truc2[3].at({i}));
  }

  for (size_t i = 0; i < 11; i++) {
    for (size_t j = 0; j < 2; j++) {
      EXPECT_EQ(full[1].at({i, j}), truc2[1].at({i, j}));
    }
  }
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 13; j++) {
      EXPECT_EQ(full[2].at({i, j}), truc2[2].at({i, j}));
    }
  }
}

TEST_F(linalg_Test, gpu_DenseUt_Pow) {
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

TEST_F(linalg_Test, gpu_DenseUt_Pow_) {
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

TEST_F(linalg_Test, gpu_DenseUt_Mod) {
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

TEST_F(linalg_Test, gpu_Tensor_Gemm) {
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

TEST_F(linalg_Test, gpu_Tensor_Gemm_) {
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
