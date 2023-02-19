#include "linalg_bk_test.h"

TEST_F(linalg_bk_Test, Svd_truncate1){
  std::vector<UniTensor> res = linalg::Svd_truncate(svd_T, 200, 0, true, true, true);
  std::vector<double> vnm_S;
  for(int i = 0; i < res[0].shape()[0];i++)
    vnm_S.push_back((float)(res[0].at({i,i}).real()));
  std::sort(vnm_S.begin(), vnm_S.end());
  for(int i = 0; i<vnm_S.size();i++)
    EXPECT_TRUE(abs(vnm_S[i]-(double)(svd_Sans.at({0,i}).real()))<1e-5);
}

TEST_F(linalg_bk_Test, Svd_truncate2){
  std::vector<UniTensor> res = linalg::Svd_truncate(svd_T, 200, 1e-1, true, true, true);
  std::vector<double> vnm_S;
  for(int i = 0; i < res[0].shape()[0];i++)
    vnm_S.push_back((float)(res[0].at({i,i}).real()));
  std::sort(vnm_S.begin(), vnm_S.end());
  for(int i = 0; i<vnm_S.size();i++)
    EXPECT_TRUE(vnm_S[i]>1e-1);
}

TEST_F(linalg_bk_Test, Qr1){
    auto res = linalg::Qr(H);
    auto Q = res[0];
    auto R = res[1];
    for(int i = 0;i<27;i++)
      for(int j = 0; j<27;j++){
          if(R.elem_exists({i,j})){
            EXPECT_TRUE(abs((double)(R.at({i,j}).real())-(double)(Qr_Rans.at({i,j}).real())) < 1E-12);
            //EXPECT_EQ((double)(R.at({i,j}).real()),(double)(Qr_Rans.at({i,j}).real()));
          }
          if(Q.elem_exists({i,j})){
            EXPECT_TRUE(abs((double)(Q.at({i,j}).real())-(double)(Qr_Qans.at({i,j}).real())) < 1E-12);
            //EXPECT_EQ((double)(Q.at({i,j}).real()),(double)(Qr_Qans.at({i,j}).real()));
          }
      }
}