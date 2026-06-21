#include <cmath>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "tn_algo_test/tfim_mpo.hpp"

using namespace cytnx;
using namespace cytnx::tn_algo;
using namespace testing;

namespace MPOTest {

  // Contract a matrix-product operator into its dense 2^N x 2^N matrix, closing
  // the open virtual bonds with the boundary vectors (e_0 on the left, e_{Dw-1}
  // on the right) used by the tn_algo DMRG driver.
  static Tensor MpoToDense(const std::vector<UniTensor> &Ws) {
    int N = static_cast<int>(Ws.size());
    cytnx_uint64 Dw = Ws[0].shape()[0];

    auto Lv = zeros({Dw});
    Lv.at<double>({0}) = 1.0;
    auto Rv = zeros({Dw});
    Rv.at<double>({Dw - 1}) = 1.0;

    UniTensor res = UniTensor(Lv, false, 0).relabel({"v0"});
    for (int k = 0; k < N; k++) {
      auto w = Ws[k].relabel({"v" + std::to_string(k), "v" + std::to_string(k + 1),
                              "o" + std::to_string(k), "i" + std::to_string(k)});
      res = Contract(res, w);
    }
    res = Contract(res, UniTensor(Rv, false, 0).relabel({"v" + std::to_string(N)}));

    std::vector<std::string> order;
    for (int k = 0; k < N; k++) order.push_back("o" + std::to_string(k));
    for (int k = 0; k < N; k++) order.push_back("i" + std::to_string(k));
    res = res.permute(order);

    auto blk = res.get_block_();
    blk.contiguous_();
    cytnx_uint64 dim = 1;
    for (int i = 0; i < N; i++) dim *= 2;
    blk.reshape_({dim, dim});
    return blk;
  }

  static bool TensorsClose(const Tensor &a, const Tensor &b, double tol = 1e-10) {
    if (a.shape() != b.shape()) return false;
    auto diff = (a - b).Norm().item<double>();
    return diff < tol;
  }

  static MPO BuildTfimMPO(int N, double J, double h) {
    MPO mpo;
    auto W = TfimTest::TfimW(J, h);
    for (int i = 0; i < N; i++) mpo.append(W);
    return mpo;
  }

  TEST(MPO, AppendAndAccess) {
    MPO mpo = BuildTfimMPO(5, 1.0, 0.5);
    EXPECT_EQ(mpo.size(), 5u);
    EXPECT_EQ(mpo.get_all().size(), 5u);
    for (cytnx_uint64 i = 0; i < mpo.size(); i++) {
      EXPECT_EQ(mpo.get_op(i).rank(), 4u);
      EXPECT_EQ(mpo.get_op(i).shape()[0], 3u);
      EXPECT_EQ(mpo.get_op(i).shape()[1], 3u);
      EXPECT_EQ(mpo.get_op(i).shape()[2], 2u);
      EXPECT_EQ(mpo.get_op(i).shape()[3], 2u);
    }
  }

  TEST(MPO, GetOpOutOfBoundThrows) {
    MPO mpo = BuildTfimMPO(3, 1.0, 0.5);
    EXPECT_ANY_THROW({ mpo.get_op(3); });
  }

  TEST(MPO, Assign) {
    MPO mpo;
    auto W = TfimTest::TfimW(1.0, 0.5);
    mpo.assign(4, W);
    EXPECT_EQ(mpo.size(), 4u);
  }

  // The MPO must encode exactly the intended Hamiltonian: contracting the chain
  // with the boundary vectors has to reproduce the dense matrix assembled
  // independently from Pauli operators.
  TEST(MPO, ContractsToCorrectHamiltonian) {
    for (int N : {2, 3, 4}) {
      double J = 1.0, h = 0.5;
      auto mpo = BuildTfimMPO(N, J, h);
      auto H_mpo = MpoToDense(mpo.get_all());
      auto H_ref = TfimTest::DenseH(N, J, h);
      EXPECT_TRUE(TensorsClose(H_mpo, H_ref))
        << "MPO-contracted Hamiltonian differs from reference for N=" << N;
    }
  }

}  // namespace MPOTest
