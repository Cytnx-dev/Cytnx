#ifndef CYTNX_TESTS_TN_ALGO_TFIM_MPO_H_
#define CYTNX_TESTS_TN_ALGO_TFIM_MPO_H_

#include <vector>
#include <string>

#include "cytnx.hpp"

// Shared helpers for the tn_algo tests built around the open-boundary
// transverse-field Ising model (TFIM)
//
//     H = -coupling * sum_i Z_i Z_{i+1} - field * sum_i X_i.
//
// The matrix-product operator is encoded in the boundary convention used by
// cytnx::tn_algo::DMRG_impl::initialize(), which fixes the left boundary vector
// to the first virtual index and the right boundary vector to the last virtual
// index. With the bond-dimension-3 upper-triangular form below, the same bulk
// tensor can be reused at every site:
//
//     W[vL, vR] =  [ I        Z            -field*X ]
//                  [ 0        0            -coupling*Z ]
//                  [ 0        0             I  ]
//
// so that  e_0 * W * W * ... * W * e_2  reproduces H.
namespace TfimTest {

  // Single-site spin-1/2 operators as 2x2 dense Tensors (Z and X are real and
  // symmetric, so the operator in/out leg order is irrelevant for this model).
  inline cytnx::Tensor Identity() { return cytnx::eye(2); }

  inline cytnx::Tensor PauliZ() {
    auto pauli_z = cytnx::zeros({2, 2});
    pauli_z.at<double>({0, 0}) = 1.0;
    pauli_z.at<double>({1, 1}) = -1.0;
    return pauli_z;
  }

  inline cytnx::Tensor PauliX() {
    auto pauli_x = cytnx::zeros({2, 2});
    pauli_x.at<double>({0, 1}) = 1.0;
    pauli_x.at<double>({1, 0}) = 1.0;
    return pauli_x;
  }

  // The single reusable bulk MPO tensor with legs [vL, vR, phys_out, phys_in].
  inline cytnx::UniTensor MakeMpoTensor(double coupling, double field) {
    cytnx::Tensor identity = Identity();
    cytnx::Tensor pauli_z = PauliZ();
    cytnx::Tensor pauli_x = PauliX();
    cytnx::Tensor w = cytnx::zeros({3, 3, 2, 2});
    auto put = [&](cytnx::cytnx_uint64 row, cytnx::cytnx_uint64 col, const cytnx::Tensor &op) {
      for (cytnx::cytnx_uint64 out_index = 0; out_index < 2; out_index++)
        for (cytnx::cytnx_uint64 in_index = 0; in_index < 2; in_index++)
          w.at<double>({row, col, out_index, in_index}) = op.at<double>({out_index, in_index});
    };
    put(0, 0, identity);
    put(0, 1, pauli_z);
    put(0, 2, -field * pauli_x);
    put(1, 2, -coupling * pauli_z);
    put(2, 2, identity);
    return cytnx::UniTensor(w, /*is_diag=*/false, /*rowrank=*/0);
  }

  // Embed a per-site operator map into the full 2^num_sites dimensional Hilbert
  // space by a chain of Kronecker products (site 0 is the most significant index).
  inline cytnx::Tensor Embed(int num_sites,
                             const std::vector<std::pair<int, cytnx::Tensor>> &site_operators) {
    cytnx::Tensor embedded = cytnx::ones({1, 1});  // 1x1 identity seed
    for (int site = 0; site < num_sites; site++) {
      cytnx::Tensor site_operator = Identity();
      for (const auto &placement : site_operators)
        if (placement.first == site) site_operator = placement.second;
      embedded = cytnx::linalg::Kron(embedded, site_operator);
    }
    return embedded;
  }

  // Dense 2^num_sites x 2^num_sites Hamiltonian assembled directly from Pauli
  // operators.
  inline cytnx::Tensor DenseHamiltonian(int num_sites, double coupling, double field) {
    cytnx::cytnx_uint64 dimension = 1;
    for (int site = 0; site < num_sites; site++) dimension *= 2;
    cytnx::Tensor hamiltonian = cytnx::zeros({dimension, dimension});
    for (int site = 0; site + 1 < num_sites; site++) {
      hamiltonian += (-coupling) * Embed(num_sites, {{site, PauliZ()}, {site + 1, PauliZ()}});
    }
    for (int site = 0; site < num_sites; site++) {
      hamiltonian += (-field) * Embed(num_sites, {{site, PauliX()}});
    }
    return hamiltonian;
  }

  // Lowest eigenvalue of the dense Hamiltonian (exact ground-state energy).
  inline double ExactGroundEnergy(int num_sites, double coupling, double field) {
    cytnx::Tensor hamiltonian = DenseHamiltonian(num_sites, coupling, field);
    std::vector<cytnx::Tensor> eigh = cytnx::linalg::Eigh(hamiltonian, false);  // eigenvalues only
    cytnx::Tensor eigenvalues = eigh[0];
    double ground_energy = eigenvalues.at<double>({0});
    for (cytnx::cytnx_uint64 i = 1; i < eigenvalues.shape()[0]; i++)
      ground_energy = std::min(ground_energy, eigenvalues.at<double>({i}));
    return ground_energy;
  }

}  // namespace TfimTest

#endif  // CYTNX_TESTS_TN_ALGO_TFIM_MPO_H_
