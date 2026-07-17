#include "ncon_test.h"

namespace cytnx {
  namespace gpu_test {

    TEST_F(NconTest, GpuNconDefaultOrder) {
      std::vector<UniTensor> tn_list = {utdnA, utdnB, utdnC};
      std::vector<std::vector<cytnx_int64>> connect_list = {{-1, -2, 2}, {2, 1}, {1, -3}};
      UniTensor res = ncon(tn_list, connect_list, true, false);
      EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
    }

    TEST_F(NconTest, GpuNconSpecifiedOrder) {
      std::vector<UniTensor> tn_list = {utdnA, utdnB, utdnC};
      std::vector<std::vector<cytnx_int64>> connect_list = {{-1, -2, 2}, {2, 1}, {1, -3}};
      UniTensor res = ncon(tn_list, connect_list, true, true, {2, 1});
      EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
    }

    TEST_F(NconTest, GpuNconOptimalOrder) {
      std::vector<UniTensor> tn_list = {utdnA, utdnB, utdnC};
      std::vector<std::vector<cytnx_int64>> connect_list = {{-1, -2, 2}, {2, 1}, {1, -3}};
      UniTensor res = ncon(tn_list, connect_list, true, true);
      EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
    }
  }  // namespace gpu_test
}  // namespace cytnx
