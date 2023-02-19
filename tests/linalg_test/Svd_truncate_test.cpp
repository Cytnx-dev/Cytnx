#include "linalg_test.h"

TEST_F(linalg_bk_Test, Svd_truncate1){
//   std::vector<UniTensor> res = linalg::Svd_truncate(svd_T, 200, 0, true, true, true);
//   std::vector<float> vnm_S;
//   for(int i = 0; i < res[0].shape()[0];i++)
//     vnm_S.push_back((float)(res[0].at({i,i}).real()));
//   std::sort(vnm_S.begin(), vnm_S.end());
//   for(int i = 0; i<vnm_S.size();i++)
//     EXPECT_TRUE(abs(vnm_S[i]-(double)(svd_Sans.at({0,i}).real()))<1e-5);
}

TEST_F(linalg_bk_Test, Svd_truncate2){
//   std::vector<UniTensor> res = linalg::Svd_truncate(svd_T, 200, 1e-1, true, true, true);
//   std::vector<float> vnm_S;
//   for(int i = 0; i < res[0].shape()[0];i++)
//     vnm_S.push_back((float)(res[0].at({i,i}).real()));
//   std::sort(vnm_S.begin(), vnm_S.end());
//   for(int i = 0; i<vnm_S.size();i++)
//     EXPECT_TRUE(abs(vnm_S[i]-(double)(svd_Sans.at({0,i}).real()))<1e-5);
}