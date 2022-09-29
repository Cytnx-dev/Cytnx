#include "Lanczos_Gnd_test.h"

class MyOp : public LinOp {
 public:
  MyOp() : LinOp("mv", 4) {}

  UniTensor matvec(const UniTensor &v) override {
    Tensor tA = arange(16).reshape(4, 4);
    UniTensor A = UniTensor(tA);
    // A += A.permute(1, 0);
    return UniTensor(linalg::Dot(A.get_block_(), v.get_block_()));
  }
};

TEST(Lanczos_Gnd, CompareWithScipyLanczos_Gnd) {
  cytnx_double evans = -0.6524758424985271;

  // Tensor testtmp = arange(16).reshape(4, 4);
  // std::cout<<testtmp<<std::endl;

  MyOp H = MyOp();
  Tensor tv = arange(4);
  UniTensor v = UniTensor(tv);
  std::vector<UniTensor> eigs =
    linalg::Lanczos(&H, v, "Gnd", 9.999999999999999988e-15, 10000, 1, false, true, 0, false);
  cytnx_double ev = (cytnx_double)eigs[0].get_block_()(0).item().real();
  std::cout << ev << ' ' << evans << std::endl;
  EXPECT_TRUE(std::fabs(ev - evans) < 1e-5);
}