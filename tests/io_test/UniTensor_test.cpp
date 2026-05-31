#include "io_test_tools.h"

#include "test_tools.h"

using namespace cytnx;
using namespace cytnx::IOTest;
using namespace cytnx::TestTools;

namespace {

  // Round-trip a UniTensor and check both metadata and element values survive.
  void ExpectRoundTrip(const UniTensor& ut) {
    UniTensor loaded = RoundTrip(ut);
    EXPECT_TRUE(AreEqUniTensorMeta(loaded, ut)) << "metadata mismatch after round-trip";
    EXPECT_TRUE(AreEqUniTensor(loaded, ut)) << "element mismatch after round-trip";
  }

}  // namespace

// Untagged dense UniTensor across all supported dtypes.
TEST(IOUniTensorTest, RoundTripDenseUntaggedAllDtypes) {
  for (auto dtype : dtype_list) {
    UniTensor ut =
      UniTensor({Bond(3), Bond(4), Bond(2)}, {"a", "b", "c"}, 1, dtype).set_name("dense");
    InitUniTensorUniform(ut, /*seed=*/dtype);
    ExpectRoundTrip(ut);
  }
}

// Tagged (directional, no-symmetry) dense UniTensor.
TEST(IOUniTensorTest, RoundTripDenseTagged) {
  UniTensor ut = UniTensor({Bond(3, BD_KET), Bond(3, BD_BRA)}, {"i", "j"}, 1, Type.ComplexDouble)
                   .set_name("tagged");
  ASSERT_TRUE(ut.is_tag());
  InitUniTensorUniform(ut, 21);
  ExpectRoundTrip(ut);
}

// Diagonal UniTensor (is_diag = true).
TEST(IOUniTensorTest, RoundTripDiagonal) {
  UniTensor ut =
    UniTensor({Bond(5), Bond(5)}, {"i", "j"}, 1, Type.Double, Device.cpu, /*is_diag=*/true)
      .set_name("diag");
  ASSERT_TRUE(ut.is_diag());
  InitUniTensorUniform(ut, 22);
  ExpectRoundTrip(ut);
}

// Symmetric (block) UniTensor with a single U1 symmetry, across numeric dtypes.
TEST(IOUniTensorTest, RoundTripSymmetricU1) {
  // Bool is excluded: block construction (Hstack) does not support it.
  for (auto dtype : {Type.Double, Type.ComplexDouble, Type.Float, Type.Int64}) {
    Bond bk = Bond(BD_KET, {Qs(0) >> 2, Qs(1) >> 3, Qs(-1) >> 1});
    Bond bb = Bond(BD_BRA, {Qs(0) >> 2, Qs(1) >> 3, Qs(-1) >> 1});
    UniTensor ut = UniTensor({bk, bb}, {"a", "b"}, -1, dtype).set_name("sym_u1");
    ASSERT_EQ(ut.uten_type(), UTenType.Block);
    InitUniTensorUniform(ut, 23);
    ExpectRoundTrip(ut);
  }
}

// Symmetric UniTensor with multiple symmetries (U1 x Z2).
TEST(IOUniTensorTest, RoundTripSymmetricMulti) {
  Bond bk = Bond(BD_KET, {{0, 0}, {1, 1}, {0, 1}}, {2, 3, 1}, {Symmetry::U1(), Symmetry::Zn(2)});
  Bond bb = Bond(BD_BRA, {{0, 0}, {1, 1}, {0, 1}}, {2, 3, 1}, {Symmetry::U1(), Symmetry::Zn(2)});
  UniTensor ut = UniTensor({bk, bb}, {"a", "b"}, -1, Type.ComplexDouble).set_name("sym_u1z2");
  ASSERT_EQ(ut.uten_type(), UTenType.Block);
  InitUniTensorUniform(ut, 24);
  ExpectRoundTrip(ut);
}

// Fermionic symmetric (BlockFermionic) UniTensor.
TEST(IOUniTensorTest, RoundTripFermionic) {
  Bond bk = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 3}, {Symmetry::FermionParity()});
  Bond bb = Bond(BD_OUT, {Qs(0) >> 2, Qs(1) >> 3}, {Symmetry::FermionParity()});
  UniTensor ut = UniTensor({bk, bb}, {"a", "b"}, -1, Type.ComplexDouble).set_name("ferm");
  ASSERT_EQ(ut.uten_type(), UTenType.BlockFermionic);
  InitUniTensorUniform(ut, 25);
  ExpectRoundTrip(ut);
}

// Saving the same name twice must fail without overwrite and succeed with it.
TEST(IOUniTensorTest, Overwrite) {
  TempH5File tmp;
  UniTensor a = UniTensor({Bond(3), Bond(2)}, {"a", "b"}, 1, Type.Double).set_name("A");
  UniTensor b =
    UniTensor({Bond(4), Bond(5), Bond(2)}, {"x", "y", "z"}, 2, Type.Double).set_name("B");
  InitUniTensorUniform(a, 1);
  InitUniTensorUniform(b, 2);
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(a, file, "obj");
    EXPECT_THROW(io::Save(b, file, "obj", "", false), std::logic_error);
    EXPECT_NO_THROW(io::Save(b, file, "obj", "", true));
    file.close();
  }
  UniTensor loaded = LoadFromFile<UniTensor>(tmp.str(), "obj");
  EXPECT_TRUE(AreEqUniTensorMeta(loaded, b));
  EXPECT_TRUE(AreEqUniTensor(loaded, b));
}

// Overwrite with identical structure but different values replaces the data.
TEST(IOUniTensorTest, OverwriteSameStructure) {
  TempH5File tmp;
  UniTensor a = UniTensor({Bond(3), Bond(2)}, {"a", "b"}, 1, Type.Double).set_name("A");
  UniTensor b = UniTensor({Bond(3), Bond(2)}, {"a", "b"}, 1, Type.Double).set_name("A");
  InitUniTensorUniform(a, 1);
  InitUniTensorUniform(b, 77);
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(a, file, "obj");
    EXPECT_THROW(io::Save(b, file, "obj", "", false), std::logic_error);
    EXPECT_NO_THROW(io::Save(b, file, "obj", "", true));
    file.close();
  }
  UniTensor loaded = LoadFromFile<UniTensor>(tmp.str(), "obj");
  EXPECT_TRUE(AreEqUniTensorMeta(loaded, b));
  EXPECT_TRUE(AreEqUniTensor(loaded, b));
}

// Store under a nested group path and read it back.
TEST(IOUniTensorTest, NestedPath) {
  TempH5File tmp;
  UniTensor ut = UniTensor({Bond(3), Bond(3)}, {"a", "b"}, 1, Type.Double).set_name("nested");
  InitUniTensorUniform(ut, 31);
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(ut, file, "obj", "/group/sub");
    file.close();
  }
  UniTensor loaded = LoadFromFile<UniTensor>(tmp.str(), "obj", "/group/sub");
  EXPECT_TRUE(AreEqUniTensorMeta(loaded, ut));
  EXPECT_TRUE(AreEqUniTensor(loaded, ut));
}

// Multiple UniTensors coexist in one file under different names.
TEST(IOUniTensorTest, MultipleObjectsInOneFile) {
  TempH5File tmp;
  UniTensor u1 = UniTensor({Bond(2), Bond(3)}, {"a", "b"}, 1, Type.Double).set_name("u1");
  UniTensor u2 = UniTensor({Bond(4), Bond(2)}, {"c", "d"}, 1, Type.ComplexDouble).set_name("u2");
  InitUniTensorUniform(u1, 41);
  InitUniTensorUniform(u2, 42);
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(u1, file, "first");
    io::Save(u2, file, "second");
    file.close();
  }
  EXPECT_TRUE(AreEqUniTensor(LoadFromFile<UniTensor>(tmp.str(), "first"), u1));
  EXPECT_TRUE(AreEqUniTensor(LoadFromFile<UniTensor>(tmp.str(), "second"), u2));
}

// The device-restoring Load overload keeps CPU data on the CPU either way.
TEST(IOUniTensorTest, LoadDeviceRestore) {
  TempH5File tmp;
  UniTensor ut = UniTensor({Bond(3), Bond(4)}, {"a", "b"}, 1, Type.Double).set_name("dev");
  InitUniTensorUniform(ut, 51);
  SaveToFile(ut, tmp.str());

  UniTensor keep = LoadFromFileDevice<UniTensor>(tmp.str(), true);
  EXPECT_EQ(keep.device(), Device.cpu);
  EXPECT_TRUE(AreEqUniTensor(keep, ut));

  UniTensor cpu = LoadFromFileDevice<UniTensor>(tmp.str(), false);
  EXPECT_EQ(cpu.device(), Device.cpu);
  EXPECT_TRUE(AreEqUniTensor(cpu, ut));
}

// Load committed reference files (regression coverage for the on-disk format).
TEST(IOUniTensorTest, LoadReference) {
  struct Ref {
    std::string file;
    UniTensor expected;
  };
  std::vector<Ref> refs = {
    {"unitensor_dense.h5", ref::unitensor_dense()},
    {"unitensor_diag.h5", ref::unitensor_diag()},
    {"unitensor_sym.h5", ref::unitensor_sym()},
    {"unitensor_fermionic.h5", ref::unitensor_fermionic()},
  };
  for (const auto& r : refs) {
    UniTensor loaded = LoadFromFile<UniTensor>(ref_data_dir() + r.file);
    EXPECT_TRUE(AreEqUniTensorMeta(loaded, r.expected)) << r.file;
    EXPECT_TRUE(AreEqUniTensor(loaded, r.expected)) << r.file;
  }
}
