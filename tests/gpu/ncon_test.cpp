#include "ncon_test.h"

using namespace std;
using namespace cytnx;

TEST_F(NconTest, gpu_ncon_default_order) {
  std::vector<UniTensor> tn_list = {utdnA, utdnB, utdnC};
  std::vector<std::vector<cytnx_int64>> connect_list = {{-1, -2, 2}, {2, 1}, {1, -3}};
  UniTensor res = ncon(tn_list, connect_list, true, false);
  EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
}

TEST_F(NconTest, gpu_ncon_specified_order) {
  std::vector<UniTensor> tn_list = {utdnA, utdnB, utdnC};
  std::vector<std::vector<cytnx_int64>> connect_list = {{-1, -2, 2}, {2, 1}, {1, -3}};
  UniTensor res = ncon(tn_list, connect_list, true, true, {2, 1});
  EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
}

TEST_F(NconTest, gpu_ncon_optimal_order) {
  std::vector<UniTensor> tn_list = {utdnA, utdnB, utdnC};
  std::vector<std::vector<cytnx_int64>> connect_list = {{-1, -2, 2}, {2, 1}, {1, -3}};
  UniTensor res = ncon(tn_list, connect_list, true, true);
  EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
}
