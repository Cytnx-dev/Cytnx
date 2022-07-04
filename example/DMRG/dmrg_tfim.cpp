#include "cytnx.hpp"

using namespace std;
using namespace cytnx;

Scalar run_DMRG(tn_algo::MPO &mpo, tn_algo::MPS &mps, int Nsweeps,
                std::vector<tn_algo::MPS> ortho_mps = {}, double weight = 40) {
  auto model = tn_algo::DMRG(mpo, mps, ortho_mps, weight);

  model.initialize();
  Scalar E;
  for (int xi = 0; xi < Nsweeps; xi++) {
    E = model.sweep();
    cout << "sweep " << xi << "/" << Nsweeps << " | Enr: " << E << endl;
  }
  return E;
}

int main() {
  int Nsites = 10;
  int chi = 16;
  double weight = 40;
  double h = 4;
  int Nsweeps = 10;

  // construct MPO:
  auto sz = cytnx::physics::pauli("z").real();
  auto sx = cytnx::physics::pauli("x").real();
  auto II = cytnx::eye(2);

  auto tM = cytnx::zeros({3, 3, 2, 2});
  tM(0, 0) = II;
  tM(-1, -1) = II;
  tM(0, 2) = -h * sx;
  tM(0, 1) = -sz;
  tM(1, 2) = sz;
  auto uM = UniTensor(tM, 0);

  auto mpo = tn_algo::MPO();
  mpo.assign(Nsites, uM);

  // starting DMRG:
  auto mps0 = tn_algo::MPS(Nsites, 2, chi);
  Scalar E0 = run_DMRG(mpo, mps0, Nsweeps);

  // first excited
  auto mps1 = tn_algo::MPS(Nsites, 2, chi);
  Scalar E1 = run_DMRG(mpo, mps1, Nsweeps, {mps0}, weight = 60);

  // second excited.
  auto mps2 = tn_algo::MPS(Nsites, 2, chi);
  Scalar E2 = run_DMRG(mpo, mps2, Nsweeps, {mps0, mps1}, weight = 60);

  cout << E0 << endl;
  cout << E1 << endl;
  cout << E2 << endl;
  return 0;
}
