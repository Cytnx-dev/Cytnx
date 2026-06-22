#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "tn_algo_test/tfim_mpo.hpp"

namespace {

  cytnx::tn_algo::MPO BuildTfimMpo(int num_sites, double coupling, double field) {
    cytnx::tn_algo::MPO mpo;
    cytnx::UniTensor bulk = TfimTest::MakeMpoTensor(coupling, field);
    for (int site = 0; site < num_sites; site++) mpo.append(bulk);
    return mpo;
  }

  // DMRG must converge to the exact ground-state energy of the dense
  // Hamiltonian. Both the MPO fed to DMRG and the reference Hamiltonian are
  // built from the same TFIM definition, so any disagreement points at the
  // tn_algo implementation rather than at a model mismatch.
  TEST(DMRG, GroundStateEnergyMatchesExact) {
    int num_sites = 6;
    double coupling = 1.0, field = 0.5;
    cytnx::cytnx_uint64 bond_dim = 16;

    cytnx::tn_algo::MPO mpo = BuildTfimMpo(num_sites, coupling, field);
    cytnx::tn_algo::MPS mps(num_sites, 2, bond_dim);

    cytnx::tn_algo::DMRG dmrg(mpo, mps);
    dmrg.initialize();

    double energy = 0.0;
    for (int sweep = 0; sweep < 8; sweep++) {
      energy = double(dmrg.sweep(/*verbose=*/false, /*maxit=*/200, /*krydim=*/4));
    }

    double exact = TfimTest::ExactGroundEnergy(num_sites, coupling, field);
    EXPECT_NEAR(energy, exact, 1e-6) << "DMRG energy " << energy << " vs exact " << exact;
  }

  // The UniTensor-based sweep (sweepv2) must reach the same ground-state energy.
  TEST(DMRG, GroundStateEnergyMatchesExactV2) {
    int num_sites = 6;
    double coupling = 1.0, field = 0.5;
    cytnx::cytnx_uint64 bond_dim = 16;

    cytnx::tn_algo::MPO mpo = BuildTfimMpo(num_sites, coupling, field);
    cytnx::tn_algo::MPS mps(num_sites, 2, bond_dim);

    cytnx::tn_algo::DMRG dmrg(mpo, mps);
    dmrg.initialize();

    double energy = 0.0;
    for (int sweep = 0; sweep < 8; sweep++) {
      energy = double(dmrg.sweepv2(/*verbose=*/false, /*maxit=*/200, /*krydim=*/4));
    }

    double exact = TfimTest::ExactGroundEnergy(num_sites, coupling, field);
    EXPECT_NEAR(energy, exact, 1e-6) << "DMRG (v2) energy " << energy << " vs exact " << exact;
  }

  // Passing a previously found state as ortho_mps must steer DMRG to the next
  // excited state instead, exercising the excited-state penalty-weight branches
  // in both sweep() and sweepv2() (the loops over ortho_mps/hLRs).
  TEST(DMRG, ExcitedStateEnergyMatchesExactSweep) {
    int num_sites = 4;
    double coupling = 0.5, field = 1.0;  // paramagnetic phase: gapped, non-degenerate
    cytnx::cytnx_uint64 bond_dim = 16;

    cytnx::tn_algo::MPO mpo = BuildTfimMpo(num_sites, coupling, field);

    cytnx::tn_algo::MPS ground_mps(num_sites, 2, bond_dim);
    cytnx::tn_algo::DMRG ground_dmrg(mpo, ground_mps);
    ground_dmrg.initialize();
    for (int sweep = 0; sweep < 8; sweep++) ground_dmrg.sweep(/*verbose=*/false, 200, 4);

    cytnx::tn_algo::MPS excited_mps(num_sites, 2, bond_dim);
    cytnx::tn_algo::DMRG excited_dmrg(mpo, excited_mps, {ground_mps}, /*weight=*/30);
    excited_dmrg.initialize();

    double energy = 0.0;
    for (int sweep = 0; sweep < 8; sweep++) {
      energy = double(excited_dmrg.sweep(/*verbose=*/false, /*maxit=*/200, /*krydim=*/4));
    }

    double exact = TfimTest::ExactFirstExcitedEnergy(num_sites, coupling, field);
    EXPECT_NEAR(energy, exact, 1e-6) << "DMRG excited energy " << energy << " vs exact " << exact;
  }

  // Same excited-state check through the UniTensor-based sweepv2() path.
  TEST(DMRG, ExcitedStateEnergyMatchesExactSweepV2) {
    int num_sites = 4;
    double coupling = 0.5, field = 1.0;
    cytnx::cytnx_uint64 bond_dim = 16;

    cytnx::tn_algo::MPO mpo = BuildTfimMpo(num_sites, coupling, field);

    cytnx::tn_algo::MPS ground_mps(num_sites, 2, bond_dim);
    cytnx::tn_algo::DMRG ground_dmrg(mpo, ground_mps);
    ground_dmrg.initialize();
    for (int sweep = 0; sweep < 8; sweep++) ground_dmrg.sweepv2(/*verbose=*/false, 200, 4);

    cytnx::tn_algo::MPS excited_mps(num_sites, 2, bond_dim);
    cytnx::tn_algo::DMRG excited_dmrg(mpo, excited_mps, {ground_mps}, /*weight=*/30);
    excited_dmrg.initialize();

    double energy = 0.0;
    for (int sweep = 0; sweep < 8; sweep++) {
      energy = double(excited_dmrg.sweepv2(/*verbose=*/false, /*maxit=*/200, /*krydim=*/4));
    }

    double exact = TfimTest::ExactFirstExcitedEnergy(num_sites, coupling, field);
    EXPECT_NEAR(energy, exact, 1e-6)
      << "DMRG (v2) excited energy " << energy << " vs exact " << exact;
  }

  // A second, more strongly transverse-field point to guard against accidental
  // agreement at one coupling.
  TEST(DMRG, GroundStateEnergyCriticalPoint) {
    int num_sites = 8;
    double coupling = 1.0, field = 1.0;  // critical TFIM
    cytnx::cytnx_uint64 bond_dim = 24;

    cytnx::tn_algo::MPO mpo = BuildTfimMpo(num_sites, coupling, field);
    cytnx::tn_algo::MPS mps(num_sites, 2, bond_dim);

    cytnx::tn_algo::DMRG dmrg(mpo, mps);
    dmrg.initialize();

    double energy = 0.0;
    for (int sweep = 0; sweep < 10; sweep++) {
      energy = double(dmrg.sweep(/*verbose=*/false, /*maxit=*/300, /*krydim=*/4));
    }

    double exact = TfimTest::ExactGroundEnergy(num_sites, coupling, field);
    EXPECT_NEAR(energy, exact, 1e-6) << "DMRG energy " << energy << " vs exact " << exact;
  }

}  // namespace
