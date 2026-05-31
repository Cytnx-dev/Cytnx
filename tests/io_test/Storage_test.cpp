#include "io_test_tools.h"

#include "test_tools.h"

using namespace cytnx;
using namespace cytnx::IOTest;

namespace {

  // Build a deterministic Storage of the requested dtype.
  Storage MakeStorage(unsigned int dtype, cytnx_uint64 n = 12) {
    std::vector<double> v(n);
    for (cytnx_uint64 i = 0; i < n; ++i) v[i] = static_cast<double>(i % 5);
    return Storage(v).astype(dtype);
  }

}  // namespace

// Round-trip a Storage of every supported dtype through the io module.
TEST(IOStorageTest, RoundTripAllDtypes) {
  for (auto dtype : TestTools::dtype_list) {
    Storage s = MakeStorage(dtype);
    Storage loaded = RoundTrip(s);
    EXPECT_EQ(loaded.dtype(), s.dtype()) << "dtype: " << Type.getname(dtype);
    EXPECT_TRUE(loaded == s) << "dtype: " << Type.getname(dtype);
  }
}

// Saving the same name twice must fail without overwrite and succeed with it.
TEST(IOStorageTest, Overwrite) {
  TempH5File tmp;
  Storage a = MakeStorage(Type.Double);
  Storage b = MakeStorage(Type.Double, 20);  // different size -> dataset gets recreated
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(a, file, "obj");
    EXPECT_THROW(io::Save(b, file, "obj", "", false), std::logic_error);
    EXPECT_NO_THROW(io::Save(b, file, "obj", "", true));
    file.close();
  }
  Storage loaded = LoadFromFile<Storage>(tmp.str(), "obj");
  EXPECT_TRUE(loaded == b);
}

// Overwrite with identical type and shape reuses the existing dataset.
TEST(IOStorageTest, OverwriteSameShape) {
  TempH5File tmp;
  Storage a = MakeStorage(Type.Double);
  Storage b = MakeStorage(Type.Double);
  b.fill(7.0);
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(a, file, "obj");
    EXPECT_NO_THROW(io::Save(b, file, "obj", "", true));
    file.close();
  }
  Storage loaded = LoadFromFile<Storage>(tmp.str(), "obj");
  EXPECT_TRUE(loaded == b);
}

// Overwrite must replace data stored with a different dtype (dataset recreated).
TEST(IOStorageTest, OverwriteDifferentDtype) {
  TempH5File tmp;
  Storage a = MakeStorage(Type.Double);
  Storage b = MakeStorage(Type.Int64, 7);  // different dtype and size
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(a, file, "obj");
    EXPECT_THROW(io::Save(b, file, "obj", "", false), std::logic_error);
    EXPECT_NO_THROW(io::Save(b, file, "obj", "", true));
    file.close();
  }
  Storage loaded = LoadFromFile<Storage>(tmp.str(), "obj");
  EXPECT_EQ(loaded.dtype(), Type.Int64);
  EXPECT_TRUE(loaded == b);
}

// Re-saving the identical object with overwrite=true keeps the data intact.
TEST(IOStorageTest, OverwriteIdentical) {
  TempH5File tmp;
  Storage a = MakeStorage(Type.Double);
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(a, file, "obj");
    EXPECT_NO_THROW(io::Save(a, file, "obj", "", true));
    file.close();
  }
  EXPECT_TRUE(LoadFromFile<Storage>(tmp.str(), "obj") == a);
}

// Objects can be stored under a nested group path and read back from it.
TEST(IOStorageTest, NestedPath) {
  TempH5File tmp;
  Storage s = MakeStorage(Type.Int64);
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(s, file, "obj", "/group/sub");
    file.close();
  }
  Storage loaded = LoadFromFile<Storage>(tmp.str(), "obj", "/group/sub");
  EXPECT_TRUE(loaded == s);
}

// Several independent objects can live in the same file under different names.
TEST(IOStorageTest, MultipleObjectsInOneFile) {
  TempH5File tmp;
  Storage s1 = MakeStorage(Type.Double);
  Storage s2 = MakeStorage(Type.ComplexDouble, 8);
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(s1, file, "first");
    io::Save(s2, file, "second");
    file.close();
  }
  Storage l1 = LoadFromFile<Storage>(tmp.str(), "first");
  Storage l2 = LoadFromFile<Storage>(tmp.str(), "second");
  EXPECT_TRUE(l1 == s1);
  EXPECT_TRUE(l2 == s2);
}

// The device-restoring Load overload keeps CPU data on the CPU either way.
TEST(IOStorageTest, LoadDeviceRestore) {
  TempH5File tmp;
  Storage s = MakeStorage(Type.Float);
  SaveToFile(s, tmp.str());

  Storage keep = LoadFromFileDevice<Storage>(tmp.str(), true);
  EXPECT_EQ(keep.device(), Device.cpu);
  EXPECT_TRUE(keep == s);

  Storage cpu = LoadFromFileDevice<Storage>(tmp.str(), false);
  EXPECT_EQ(cpu.device(), Device.cpu);
  EXPECT_TRUE(cpu == s);
}

// Load committed reference file (regression coverage for the on-disk format).
TEST(IOStorageTest, LoadReference) {
  Storage loaded = LoadFromFile<Storage>(ref_data_dir() + "storage.h5");
  Storage expected = ref::storage();
  EXPECT_EQ(loaded.dtype(), expected.dtype());
  EXPECT_TRUE(loaded == expected);
}
