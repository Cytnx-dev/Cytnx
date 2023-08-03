#include "Lanczos_Gnd_test.h"

class MyOp : public LinOp {
 public:
  MyOp() : LinOp("mv", 4) {}

  UniTensor matvec(const UniTensor& v) override {
    Tensor tA = arange(27 * 27).reshape(27, 27);
    UniTensor A = UniTensor(tA);
    // A = A + A.clone().permute({1, 0},-1,false);
    A = A + A.Transpose();
    return UniTensor(linalg::Dot(A.get_block_(), v.get_block_()));
  }
};

class MyOp2 : public LinOp {
 public:
  UniTensor H;
  MyOp2(int dim) : LinOp("mv", dim) {
    Tensor A = Tensor::Load("../../tests/test_data_base/linalg/Lanczos_Gnd/lan_block_A.cytn");
    Tensor B = Tensor::Load("../../tests/test_data_base/linalg/Lanczos_Gnd/lan_block_B.cytn");
    Tensor C = Tensor::Load("../../tests/test_data_base/linalg/Lanczos_Gnd/lan_block_C.cytn");
    Bond lan_I = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
    Bond lan_J = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
    H = UniTensor({lan_I, lan_J});
    H.put_block(A, 0);
    H.put_block(B, 1);
    H.put_block(C, 2);
    H.set_labels({"a", "b"});
    // H.print_diagram();
    // H.print_blocks();
  }
  UniTensor matvec(const UniTensor& psi) override {
    auto out = H.contract(psi);
    out.set_labels({"b", "c"});
    return out;
  }
};

TEST(Lanczos_Gnd, Lanczos_Gnd_test) {
  // CompareWithScipy

  // cytnx_double evans = -0.6524758424985271;
  cytnx_double evans = -1628.9964650426593;

  // Tensor testtmp = arange(16).reshape(4, 4);
  // std::cout<<testtmp<<std::endl;
  MyOp H = MyOp();
  Tensor tv = arange(27);
  UniTensor v = UniTensor(tv);
  std::vector<UniTensor> eigs =
    linalg::Lanczos(&H, v, "Gnd", 9.999999999999999988e-15, 10000, 1, false, true, 0, false);
  cytnx_double ev = (cytnx_double)eigs[0].get_block_()(0).item().real();
  std::cout << ev << ' ' << evans << std::endl;
  EXPECT_TRUE(std::fabs(ev - evans) < 1e-5);
  // EXPECT_DOUBLE_EQ(ev, evans);
}

TEST(Lanczos_Gnd, Bk_Lanczos_Gnd_test) {
  // CompareWithScipy
  cytnx_double evans = -2.31950925;

  Bond lan_I_v = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
  Bond lan_J_v = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {1, 1, 1});
  UniTensor lan_guess = UniTensor({lan_I_v, lan_J_v});

  lan_guess.put_block(random::normal(9, 1, 1).reshape({9, 1}), 0);
  lan_guess.put_block(random::normal(9, 1, 1).reshape({9, 1}), 1);
  lan_guess.put_block(random::normal(9, 1, 1).reshape({9, 1}), 2);
  lan_guess.set_labels({"b", "c"});
  // lan_guess.print_diagram();
  // lan_guess.print_blocks();

  MyOp2 H = MyOp2(27);

  std::vector<UniTensor> eigs =
    linalg::Lanczos(&H, lan_guess, "Gnd", 9.999999999999999988e-15, 10000, 1, false, true, 0, true);
  cytnx_double ev = (cytnx_double)eigs[0].get_block_()(0).item().real();
  std::cout << ev << ' ' << evans << std::endl;
  EXPECT_TRUE(std::fabs(ev - evans) < 1e-5);
  // EXPECT_DOUBLE_EQ(ev, evans);
}
