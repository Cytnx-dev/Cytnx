#include <cmath>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "tn_algo_test/tfim_mpo.hpp"

using namespace cytnx;
using namespace cytnx::tn_algo;

namespace {

  // Contract a matrix-product operator into its dense 2^num_sites matrix, closing
  // the open virtual bonds with the boundary vectors (e_0 on the left,
  // e_{bond_dim-1} on the right) used by the tn_algo DMRG driver.
  Tensor MpoToDense(const std::vector<UniTensor> &mpo_tensors) {
    int num_sites = static_cast<int>(mpo_tensors.size());
    cytnx_uint64 bond_dim = mpo_tensors[0].shape()[0];

    Tensor left_boundary = zeros({bond_dim});
    left_boundary.at<double>({0}) = 1.0;
    Tensor right_boundary = zeros({bond_dim});
    right_boundary.at<double>({bond_dim - 1}) = 1.0;

    UniTensor contracted =
      UniTensor(left_boundary, /*is_diag=*/false, /*rowrank=*/0).relabel({"v0"});
    for (int site = 0; site < num_sites; site++) {
      UniTensor w =
        mpo_tensors[site].relabel({"v" + std::to_string(site), "v" + std::to_string(site + 1),
                                   "o" + std::to_string(site), "i" + std::to_string(site)});
      contracted = Contract(contracted, w);
    }
    contracted = Contract(contracted, UniTensor(right_boundary, /*is_diag=*/false, /*rowrank=*/0)
                                        .relabel({"v" + std::to_string(num_sites)}));

    std::vector<std::string> order;
    for (int site = 0; site < num_sites; site++) order.push_back("o" + std::to_string(site));
    for (int site = 0; site < num_sites; site++) order.push_back("i" + std::to_string(site));
    contracted = contracted.permute(order);

    Tensor block = contracted.get_block_();
    block.contiguous_();
    cytnx_uint64 dimension = 1;
    for (int site = 0; site < num_sites; site++) dimension *= 2;
    block.reshape_({dimension, dimension});
    return block;
  }

  bool TensorsClose(const Tensor &lhs, const Tensor &rhs, double tol = 1e-10) {
    if (lhs.shape() != rhs.shape()) return false;
    return (lhs - rhs).Norm().item<double>() < tol;
  }

  MPO BuildTfimMpo(int num_sites, double coupling, double field) {
    MPO mpo;
    UniTensor bulk = TfimTest::MakeMpoTensor(coupling, field);
    for (int site = 0; site < num_sites; site++) mpo.append(bulk);
    return mpo;
  }

  TEST(MPO, AppendAndAccess) {
    MPO mpo = BuildTfimMpo(5, 1.0, 0.5);
    EXPECT_EQ(mpo.size(), 5u);
    EXPECT_EQ(mpo.get_all().size(), 5u);
    for (cytnx_uint64 site = 0; site < mpo.size(); site++) {
      EXPECT_EQ(mpo.get_op(site).rank(), 4u);
      EXPECT_EQ(mpo.get_op(site).shape()[0], 3u);
      EXPECT_EQ(mpo.get_op(site).shape()[1], 3u);
      EXPECT_EQ(mpo.get_op(site).shape()[2], 2u);
      EXPECT_EQ(mpo.get_op(site).shape()[3], 2u);
    }
  }

  TEST(MPO, GetOpOutOfBoundThrows) {
    MPO mpo = BuildTfimMpo(3, 1.0, 0.5);
    EXPECT_ANY_THROW({ mpo.get_op(3); });
  }

  TEST(MPO, Assign) {
    MPO mpo;
    UniTensor bulk = TfimTest::MakeMpoTensor(1.0, 0.5);
    mpo.assign(4, bulk);
    EXPECT_EQ(mpo.size(), 4u);
  }

  // The MPO must encode exactly the intended Hamiltonian: contracting the chain
  // with the boundary vectors has to reproduce the dense matrix assembled
  // independently from Pauli operators.
  TEST(MPO, ContractsToCorrectHamiltonian) {
    for (int num_sites : {2, 3, 4}) {
      double coupling = 1.0, field = 0.5;
      MPO mpo = BuildTfimMpo(num_sites, coupling, field);
      Tensor mpo_hamiltonian = MpoToDense(mpo.get_all());
      Tensor reference_hamiltonian = TfimTest::DenseHamiltonian(num_sites, coupling, field);
      EXPECT_TRUE(TensorsClose(mpo_hamiltonian, reference_hamiltonian))
        << "MPO-contracted Hamiltonian differs from reference for num_sites=" << num_sites;
    }
  }

}  // namespace
