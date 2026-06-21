#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "tn_algo_test/tfim_mpo.hpp"

using namespace cytnx;
using namespace cytnx::tn_algo;
using namespace testing;

namespace DMRGTest {

  static MPO BuildTfimMPO(int N, double J, double h) {
    MPO mpo;
    auto W = TfimTest::TfimW(J, h);
    for (int i = 0; i < N; i++) mpo.append(W);
    return mpo;
  }

  // DMRG must converge to the exact ground-state energy of the dense
  // Hamiltonian. Both the MPO fed to DMRG and the reference Hamiltonian are
  // built from the same TFIM definition, so any disagreement points at the
  // tn_algo implementation rather than at a model mismatch.
  TEST(DMRG, GroundStateEnergyMatchesExact) {
    int N = 6;
    double J = 1.0, h = 0.5;
    cytnx_uint64 chi = 16;

    auto mpo = BuildTfimMPO(N, J, h);
    MPS mps(N, 2, chi);

    auto dmrg = DMRG(mpo, mps);
    dmrg.initialize();

    double energy = 0.0;
    for (int sweep = 0; sweep < 8; sweep++) {
      energy = double(dmrg.sweep(/*verbose=*/false, /*maxit=*/200, /*krydim=*/4));
    }

    double exact = TfimTest::ExactGroundEnergy(N, J, h);
    EXPECT_NEAR(energy, exact, 1e-6) << "DMRG energy " << energy << " vs exact " << exact;
  }

  // The UniTensor-based sweep (sweepv2) must reach the same ground-state energy.
  TEST(DMRG, GroundStateEnergyMatchesExactV2) {
    int N = 6;
    double J = 1.0, h = 0.5;
    cytnx_uint64 chi = 16;

    auto mpo = BuildTfimMPO(N, J, h);
    MPS mps(N, 2, chi);

    auto dmrg = DMRG(mpo, mps);
    dmrg.initialize();

    double energy = 0.0;
    for (int sweep = 0; sweep < 8; sweep++) {
      energy = double(dmrg.sweepv2(/*verbose=*/false, /*maxit=*/200, /*krydim=*/4));
    }

    double exact = TfimTest::ExactGroundEnergy(N, J, h);
    EXPECT_NEAR(energy, exact, 1e-6) << "DMRG (v2) energy " << energy << " vs exact " << exact;
  }

  // A second, more strongly transverse-field point to guard against accidental
  // agreement at one coupling.
  TEST(DMRG, GroundStateEnergyCriticalPoint) {
    int N = 8;
    double J = 1.0, h = 1.0;  // critical TFIM
    cytnx_uint64 chi = 24;

    auto mpo = BuildTfimMPO(N, J, h);
    MPS mps(N, 2, chi);

    auto dmrg = DMRG(mpo, mps);
    dmrg.initialize();

    double energy = 0.0;
    for (int sweep = 0; sweep < 10; sweep++) {
      energy = double(dmrg.sweep(/*verbose=*/false, /*maxit=*/300, /*krydim=*/4));
    }

    double exact = TfimTest::ExactGroundEnergy(N, J, h);
    EXPECT_NEAR(energy, exact, 1e-6) << "DMRG energy " << energy << " vs exact " << exact;
  }

}  // namespace DMRGTest
