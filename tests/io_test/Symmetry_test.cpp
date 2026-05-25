#include "io_test_tools.h"

using namespace cytnx;
using namespace cytnx::IOTest;

// Round-trip the U1 and Zn symmetry generators.
TEST(IOSymmetryTest, RoundTripU1) {
  Symmetry s = Symmetry::U1();
  Symmetry loaded = RoundTrip(s);
  EXPECT_TRUE(loaded == s);
}

TEST(IOSymmetryTest, RoundTripZn) {
  for (int n : {2, 3, 5}) {
    Symmetry s = Symmetry::Zn(n);
    Symmetry loaded = RoundTrip(s);
    EXPECT_TRUE(loaded == s) << "Zn n=" << n;
  }
}

// Round-trip the fermionic symmetry generators.
TEST(IOSymmetryTest, RoundTripFermionic) {
  for (Symmetry s : {Symmetry::FermionParity(), Symmetry::FermionNumber()}) {
    Symmetry loaded = RoundTrip(s);
    EXPECT_TRUE(loaded == s);
  }
}

// Saving the same name twice must fail without overwrite and succeed with it.
TEST(IOSymmetryTest, Overwrite) {
  TempH5File tmp;
  Symmetry a = Symmetry::U1();
  Symmetry b = Symmetry::Zn(2);
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(a, file, "obj");
    EXPECT_THROW(io::Save(b, file, "obj", "", false), std::logic_error);
    EXPECT_NO_THROW(io::Save(b, file, "obj", "", true));
    file.close();
  }
  Symmetry loaded = LoadFromFile<Symmetry>(tmp.str(), "obj");
  EXPECT_TRUE(loaded == b);
}

// Store under a nested group path and read it back.
TEST(IOSymmetryTest, NestedPath) {
  TempH5File tmp;
  Symmetry s = Symmetry::Zn(4);
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(s, file, "obj", "/sym/inner");
    file.close();
  }
  Symmetry loaded = LoadFromFile<Symmetry>(tmp.str(), "obj", "/sym/inner");
  EXPECT_TRUE(loaded == s);
}

// Multiple symmetries coexist in one file under different names.
TEST(IOSymmetryTest, MultipleObjectsInOneFile) {
  TempH5File tmp;
  Symmetry s1 = Symmetry::U1();
  Symmetry s2 = Symmetry::Zn(3);
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(s1, file, "first");
    io::Save(s2, file, "second");
    file.close();
  }
  EXPECT_TRUE(LoadFromFile<Symmetry>(tmp.str(), "first") == s1);
  EXPECT_TRUE(LoadFromFile<Symmetry>(tmp.str(), "second") == s2);
}

// Load committed reference files (regression coverage for the on-disk format).
TEST(IOSymmetryTest, LoadReference) {
  EXPECT_TRUE(LoadFromFile<Symmetry>(ref_data_dir() + "symmetry.h5") == ref::symmetry());
  EXPECT_TRUE(LoadFromFile<Symmetry>(ref_data_dir() + "symmetry_fpar.h5") == ref::symmetry_fpar());
}
