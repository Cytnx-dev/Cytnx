#ifndef CYTNX_TESTS_TN_ALGO_TFIM_MPO_H_
#define CYTNX_TESTS_TN_ALGO_TFIM_MPO_H_

#include <vector>
#include <string>

#include "cytnx.hpp"

// Shared helpers for the tn_algo tests built around the open-boundary
// transverse-field Ising model (TFIM)
//
//     H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i.
//
// The matrix-product operator is encoded in the boundary convention used by
// cytnx::tn_algo::DMRG_impl::initialize(), which fixes the left boundary vector
// to the first virtual index and the right boundary vector to the last virtual
// index. With the bond-dimension-3 upper-triangular form below, the same bulk
// tensor can be reused at every site:
//
//     W[vL, vR] =  [ I    Z    -hX ]
//                  [ 0    0    -JZ ]
//                  [ 0    0     I  ]
//
// so that  e_0 * W * W * ... * W * e_2  reproduces H.
namespace TfimTest {

  // Single-site spin-1/2 operators as 2x2 dense Tensors (Z and X are real and
  // symmetric, so the operator in/out leg order is irrelevant for this model).
  inline cytnx::Tensor PauliI() { return cytnx::eye(2); }

  inline cytnx::Tensor PauliZ() {
    auto Z = cytnx::zeros({2, 2});
    Z.at<double>({0, 0}) = 1.0;
    Z.at<double>({1, 1}) = -1.0;
    return Z;
  }

  inline cytnx::Tensor PauliX() {
    auto X = cytnx::zeros({2, 2});
    X.at<double>({0, 1}) = 1.0;
    X.at<double>({1, 0}) = 1.0;
    return X;
  }

  // The single reusable bulk MPO tensor with legs [vL, vR, phys_out, phys_in].
  inline cytnx::UniTensor TfimW(double J, double h) {
    using namespace cytnx;
    auto I = PauliI();
    auto Z = PauliZ();
    auto X = PauliX();
    auto W = zeros({3, 3, 2, 2});
    auto put = [&](cytnx_uint64 a, cytnx_uint64 b, const Tensor& op) {
      for (cytnx_uint64 s = 0; s < 2; s++)
        for (cytnx_uint64 t = 0; t < 2; t++) W.at<double>({a, b, s, t}) = op.at<double>({s, t});
    };
    put(0, 0, I);
    put(0, 1, Z);
    put(0, 2, -h * X);
    put(1, 2, -J * Z);
    put(2, 2, I);
    return UniTensor(W, false, 0);
  }

  // Embed a per-site operator map into the full 2^N dimensional Hilbert space by
  // a chain of Kronecker products (site 0 is the most significant index).
  inline cytnx::Tensor Embed(int N, const std::vector<std::pair<int, cytnx::Tensor>>& ops) {
    using namespace cytnx;
    Tensor acc = ones({1, 1});  // 1x1 identity seed
    for (int site = 0; site < N; site++) {
      Tensor op = PauliI();
      for (const auto& kv : ops)
        if (kv.first == site) op = kv.second;
      acc = linalg::Kron(acc, op);
    }
    return acc;
  }

  // Dense 2^N x 2^N Hamiltonian assembled directly from Pauli operators.
  inline cytnx::Tensor DenseH(int N, double J, double h) {
    using namespace cytnx;
    cytnx_uint64 dim = 1;
    for (int i = 0; i < N; i++) dim *= 2;
    Tensor H = zeros({dim, dim});
    for (int i = 0; i + 1 < N; i++) {
      H += (-J) * Embed(N, {{i, PauliZ()}, {i + 1, PauliZ()}});
    }
    for (int i = 0; i < N; i++) {
      H += (-h) * Embed(N, {{i, PauliX()}});
    }
    return H;
  }

  // Lowest eigenvalue of the dense Hamiltonian (exact ground-state energy).
  inline double ExactGroundEnergy(int N, double J, double h) {
    using namespace cytnx;
    auto H = DenseH(N, J, h);
    auto eig = linalg::Eigh(H, false);  // eigenvalues only
    auto evals = eig[0];
    double gs = evals.at<double>({0});
    for (cytnx_uint64 i = 1; i < evals.shape()[0]; i++) gs = std::min(gs, evals.at<double>({i}));
    return gs;
  }

}  // namespace TfimTest

#endif  // CYTNX_TESTS_TN_ALGO_TFIM_MPO_H_
