#include "io_test_tools.h"

using namespace cytnx;
using namespace cytnx::IOTest;

// A plain (untagged, no-symmetry) bond.
TEST(IOBondTest, RoundTripRegular) {
  Bond b = Bond(7);
  Bond loaded = RoundTrip(b);
  EXPECT_TRUE(loaded == b);
}

// Directional (tagged) bonds without symmetry.
TEST(IOBondTest, RoundTripDirectional) {
  for (auto bt : {BD_IN, BD_OUT}) {
    Bond b = Bond(5, bt);
    Bond loaded = RoundTrip(b);
    EXPECT_TRUE(loaded == b);
  }
}

// A single-U1-symmetry bond.
TEST(IOBondTest, RoundTripU1) {
  Bond b = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 2, Qs(-1) >> 3});
  Bond loaded = RoundTrip(b);
  EXPECT_TRUE(loaded == b);
}

// A bond carrying multiple symmetries (U1 x Z2).
TEST(IOBondTest, RoundTripMultiSymmetry) {
  Bond b =
    Bond(BD_KET, {{0, 2}, {1, 5}, {1, 6}, {0, 1}}, {4, 7, 2, 3}, {Symmetry::Zn(2), Symmetry::U1()});
  Bond loaded = RoundTrip(b);
  EXPECT_TRUE(loaded == b);
}

// Bonds carrying a fermionic-parity symmetry.
TEST(IOBondTest, RoundTripFermionic) {
  Bond b = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 3}, {Symmetry::FermionParity()});
  Bond loaded = RoundTrip(b);
  EXPECT_TRUE(loaded == b);
}

// Saving the same name twice must fail without overwrite and succeed with it.
TEST(IOBondTest, Overwrite) {
  TempH5File tmp;
  Bond a = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 2});
  Bond b = Bond(BD_OUT, {Qs(0) >> 2, Qs(2) >> 3, Qs(-1) >> 1});
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(a, file, "obj");
    EXPECT_THROW(io::Save(b, file, "obj", "", false), std::logic_error);
    EXPECT_NO_THROW(io::Save(b, file, "obj", "", true));
    file.close();
  }
  Bond loaded = LoadFromFile<Bond>(tmp.str(), "obj");
  EXPECT_TRUE(loaded == b);
}

// Store under a nested group path and read it back.
TEST(IOBondTest, NestedPath) {
  TempH5File tmp;
  Bond b = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 2});
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(b, file, "obj", "/bonds/a");
    file.close();
  }
  Bond loaded = LoadFromFile<Bond>(tmp.str(), "obj", "/bonds/a");
  EXPECT_TRUE(loaded == b);
}

// Multiple bonds coexist in one file under different names.
TEST(IOBondTest, MultipleObjectsInOneFile) {
  TempH5File tmp;
  Bond b1 = Bond(4);
  Bond b2 = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1});
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(b1, file, "first");
    io::Save(b2, file, "second");
    file.close();
  }
  EXPECT_TRUE(LoadFromFile<Bond>(tmp.str(), "first") == b1);
  EXPECT_TRUE(LoadFromFile<Bond>(tmp.str(), "second") == b2);
}

// Load committed reference files (regression coverage for the on-disk format).
TEST(IOBondTest, LoadReference) {
  EXPECT_TRUE(LoadFromFile<Bond>(ref_data_dir() + "bond.h5") == ref::bond());
  EXPECT_TRUE(LoadFromFile<Bond>(ref_data_dir() + "bond_fermionic.h5") == ref::bond_fermionic());
}
