#include "io_test_tools.h"

#include "test_tools.h"

using namespace cytnx;
using namespace cytnx::IOTest;
using namespace cytnx::TestTools;

// Round-trip a Tensor of every supported dtype.
TEST(IOTensorTest, RoundTripAllDtypes) {
  for (auto dtype : dtype_list) {
    Tensor t = Tensor({3, 4, 2}, dtype);
    InitTensorUniform(t, /*seed=*/dtype);
    Tensor loaded = RoundTrip(t);
    EXPECT_TRUE(AreEqTensor(loaded, t)) << "dtype: " << Type.getname(dtype);
  }
}

// Round-trip tensors of several ranks/shapes.
TEST(IOTensorTest, RoundTripShapes) {
  std::vector<std::vector<cytnx_uint64>> shapes = {{6}, {3, 4}, {2, 3, 4}, {2, 2, 2, 2}};
  for (const auto& shape : shapes) {
    Tensor t = Tensor(shape, Type.Double);
    InitTensorUniform(t, 7);
    Tensor loaded = RoundTrip(t);
    EXPECT_TRUE(AreEqTensor(loaded, t));
  }
}

// A non-contiguous (permuted) tensor must round-trip to its logical (contiguous) values.
TEST(IOTensorTest, RoundTripNonContiguous) {
  Tensor t = Tensor({3, 4, 5}, Type.ComplexDouble);
  InitTensorUniform(t, 11);
  t.permute_({2, 0, 1});
  ASSERT_FALSE(t.is_contiguous());
  Tensor loaded = RoundTrip(t);
  EXPECT_TRUE(loaded.is_contiguous());
  EXPECT_TRUE(AreEqTensor(loaded, t.contiguous()));
}

// Saving the same name twice must fail without overwrite and succeed with it.
TEST(IOTensorTest, Overwrite) {
  TempH5File tmp;
  Tensor a = Tensor({3, 4}, Type.Double);
  InitTensorUniform(a, 1);
  Tensor b = Tensor({2, 5}, Type.Double);  // different shape -> dataset recreated
  InitTensorUniform(b, 2);
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(a, file, "obj");
    EXPECT_THROW(io::Save(b, file, "obj", "", false), std::logic_error);
    EXPECT_NO_THROW(io::Save(b, file, "obj", "", true));
    file.close();
  }
  Tensor loaded = LoadFromFile<Tensor>(tmp.str(), "obj");
  EXPECT_TRUE(AreEqTensor(loaded, b));
}

// Overwrite with identical shape/dtype but different values replaces the data.
TEST(IOTensorTest, OverwriteSameShape) {
  TempH5File tmp;
  Tensor a = Tensor({3, 4}, Type.Double);
  Tensor b = Tensor({3, 4}, Type.Double);
  InitTensorUniform(a, 1);
  InitTensorUniform(b, 99);
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(a, file, "obj");
    EXPECT_THROW(io::Save(b, file, "obj", "", false), std::logic_error);
    EXPECT_NO_THROW(io::Save(b, file, "obj", "", true));
    file.close();
  }
  EXPECT_TRUE(AreEqTensor(LoadFromFile<Tensor>(tmp.str(), "obj"), b));
}

// Overwrite must replace data stored with a different dtype (dataset recreated).
TEST(IOTensorTest, OverwriteDifferentDtype) {
  TempH5File tmp;
  Tensor a = Tensor({3, 4}, Type.Double);
  Tensor b = Tensor({3, 4}, Type.ComplexDouble);
  InitTensorUniform(a, 1);
  InitTensorUniform(b, 2);
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(a, file, "obj");
    EXPECT_NO_THROW(io::Save(b, file, "obj", "", true));
    file.close();
  }
  EXPECT_TRUE(AreEqTensor(LoadFromFile<Tensor>(tmp.str(), "obj"), b));
}

// Store under a nested group path and read it back.
TEST(IOTensorTest, NestedPath) {
  TempH5File tmp;
  Tensor t = Tensor({4, 4}, Type.Float);
  InitTensorUniform(t, 3);
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(t, file, "obj", "/deeply/nested/group");
    file.close();
  }
  Tensor loaded = LoadFromFile<Tensor>(tmp.str(), "obj", "/deeply/nested/group");
  EXPECT_TRUE(AreEqTensor(loaded, t));
}

// Multiple tensors coexist in one file under different names.
TEST(IOTensorTest, MultipleObjectsInOneFile) {
  TempH5File tmp;
  Tensor t1 = Tensor({3, 3}, Type.Double);
  Tensor t2 = Tensor({2, 4}, Type.Int32);
  InitTensorUniform(t1, 4);
  InitTensorUniform(t2, 5);
  {
    H5::H5File file = io::open(tmp.str(), io::ACC_TRUNC);
    io::Save(t1, file, "alpha");
    io::Save(t2, file, "beta");
    file.close();
  }
  EXPECT_TRUE(AreEqTensor(LoadFromFile<Tensor>(tmp.str(), "alpha"), t1));
  EXPECT_TRUE(AreEqTensor(LoadFromFile<Tensor>(tmp.str(), "beta"), t2));
}

// The device-restoring Load overload keeps CPU data on the CPU either way.
TEST(IOTensorTest, LoadDeviceRestore) {
  TempH5File tmp;
  Tensor t = Tensor({3, 4}, Type.Double);
  InitTensorUniform(t, 6);
  SaveToFile(t, tmp.str());

  Tensor keep = LoadFromFileDevice<Tensor>(tmp.str(), true);
  EXPECT_EQ(keep.device(), Device.cpu);
  EXPECT_TRUE(AreEqTensor(keep, t));

  Tensor cpu = LoadFromFileDevice<Tensor>(tmp.str(), false);
  EXPECT_EQ(cpu.device(), Device.cpu);
  EXPECT_TRUE(AreEqTensor(cpu, t));
}

// Load committed reference file (regression coverage for the on-disk format).
TEST(IOTensorTest, LoadReference) {
  Tensor loaded = LoadFromFile<Tensor>(ref_data_dir() + "tensor.h5");
  EXPECT_TRUE(AreEqTensor(loaded, ref::tensor()));
}
