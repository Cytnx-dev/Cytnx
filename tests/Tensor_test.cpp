#include "Tensor_test.h"
#include "test_tools.h"

#include <filesystem>
#include <fstream>

TEST_F(TensorTest, Constructor) {
  Tensor A;
  EXPECT_EQ(A.dtype(), Type.Void);
  EXPECT_TRUE(A.is_void());
  EXPECT_FALSE(A.is_scalar());
  EXPECT_EQ(A.device(), Device.cpu);
  EXPECT_EQ(A.shape().size(), 0);
  EXPECT_EQ(A.is_contiguous(), true);

  Tensor B({3, 4, 5});
  EXPECT_EQ(B.dtype(), Type.Double);
  EXPECT_FALSE(B.is_void());
  EXPECT_EQ(B.device(), Device.cpu);
  EXPECT_EQ(B.shape().size(), 3);
  EXPECT_EQ(B.shape()[0], 3);
  EXPECT_EQ(B.shape()[1], 4);
  EXPECT_EQ(B.shape()[2], 5);
  EXPECT_EQ(B.is_contiguous(), true);

  Tensor C({3, 4, 5}, Type.Double);
  EXPECT_EQ(C.dtype(), Type.Double);
  EXPECT_FALSE(C.is_void());
  EXPECT_EQ(C.device(), Device.cpu);
  EXPECT_EQ(C.shape().size(), 3);
  EXPECT_EQ(C.shape()[0], 3);
  EXPECT_EQ(C.shape()[1], 4);
  EXPECT_EQ(C.shape()[2], 5);
  EXPECT_EQ(C.is_contiguous(), true);

  Tensor S(std::vector<cytnx_uint64>{}, Type.Double);
  EXPECT_EQ(S.dtype(), Type.Double);
  EXPECT_FALSE(S.is_void());
  EXPECT_EQ(S.device(), Device.cpu);
  EXPECT_EQ(S.shape().size(), 0);
  EXPECT_EQ(S.rank(), 0);
  EXPECT_TRUE(S.is_scalar());
  EXPECT_EQ(S.storage().size(), 1);
  EXPECT_EQ(S.is_contiguous(), true);
#ifdef UNI_GPU
  Tensor D({3, 4, 5}, Type.Double, Device.cuda);
  EXPECT_EQ(D.dtype(), Type.Double);
  EXPECT_EQ(D.device(), Device.cuda);
  EXPECT_EQ(D.shape().size(), 3);
  EXPECT_EQ(D.shape()[0], 3);
  EXPECT_EQ(D.shape()[1], 4);
  EXPECT_EQ(D.shape()[2], 5);
  EXPECT_EQ(D.is_contiguous(), true);

  Tensor E({3, 4, 5}, Type.Double, Device.cuda, true);
  EXPECT_EQ(E.dtype(), Type.Double);
  EXPECT_EQ(E.device(), Device.cuda);
  EXPECT_EQ(E.shape().size(), 3);
  EXPECT_EQ(E.shape()[0], 3);
  EXPECT_EQ(E.shape()[1], 4);
  EXPECT_EQ(E.shape()[2], 5);
  EXPECT_EQ(E.is_contiguous(), true);

  Tensor F({3, 4, 5}, Type.Double, Device.cuda, false);
  EXPECT_EQ(F.dtype(), Type.Double);
  EXPECT_EQ(F.device(), Device.cuda);
  EXPECT_EQ(F.shape().size(), 3);
  EXPECT_EQ(F.shape()[0], 3);
  EXPECT_EQ(F.shape()[1], 4);
  EXPECT_EQ(F.shape()[2], 5);
  EXPECT_EQ(F.is_contiguous(), true);
#endif
}

TEST_F(TensorTest, VoidTensorAtEmptyLocatorThrows) {
  Tensor uninitialized;
  EXPECT_EQ(uninitialized.dtype(), Type.Void);
  EXPECT_TRUE(uninitialized.is_void());
  EXPECT_THROW(uninitialized.at<double>({}), std::logic_error);
  EXPECT_THROW(uninitialized.at({}), std::logic_error);

  const Tensor &const_uninitialized = uninitialized;
  EXPECT_THROW(const_uninitialized.at({}), std::logic_error);
}

TEST_F(TensorTest, CopyConstructor) {
#ifdef UNI_GPU
  Tensor A({3, 4, 5}, Type.Double, Device.cuda, false);
  Tensor B(A);
  EXPECT_EQ(B.dtype(), Type.Double);
  EXPECT_EQ(B.device(), Device.cuda);
  EXPECT_EQ(B.shape().size(), 3);
  EXPECT_EQ(B.shape()[0], 3);
  EXPECT_EQ(B.shape()[1], 4);
  EXPECT_EQ(B.shape()[2], 5);
  EXPECT_EQ(B.is_contiguous(), true);

  Tensor C = A;
  EXPECT_EQ(C.dtype(), Type.Double);
  EXPECT_EQ(C.device(), Device.cuda);
  EXPECT_EQ(C.shape().size(), 3);
  EXPECT_EQ(C.shape()[0], 3);
  EXPECT_EQ(C.shape()[1], 4);
  EXPECT_EQ(C.shape()[2], 5);
  EXPECT_EQ(C.is_contiguous(), true);
#endif

  Tensor D;
  D = tarcomplex3456;
  EXPECT_EQ(D.dtype(), Type.ComplexDouble);
  EXPECT_EQ(D.device(), Device.cpu);
  EXPECT_EQ(D.shape().size(), 4);
  EXPECT_EQ(D.shape()[0], 3);
  EXPECT_EQ(D.shape()[1], 4);
  EXPECT_EQ(D.shape()[2], 5);
  EXPECT_EQ(D.shape()[3], 6);
  EXPECT_EQ(D.is_contiguous(), true);

  Tensor E;
  E = tarcomplex3456.permute({1, 2, 3, 0});
  EXPECT_EQ(E.dtype(), Type.ComplexDouble);
  EXPECT_EQ(E.device(), Device.cpu);
  EXPECT_EQ(E.shape().size(), 4);
  EXPECT_EQ(E.shape()[0], 4);
  EXPECT_EQ(E.shape()[1], 5);
  EXPECT_EQ(E.shape()[2], 6);
  EXPECT_EQ(E.shape()[3], 3);
  EXPECT_EQ(E.is_contiguous(), false);
}

TEST_F(TensorTest, shape) {
  Tensor A({3, 4, 5}, Type.Double, Device.cpu, false);
  EXPECT_EQ(A.shape().size(), 3);
  EXPECT_EQ(A.shape()[0], 3);
  EXPECT_EQ(A.shape()[1], 4);
  EXPECT_EQ(A.shape()[2], 5);
  EXPECT_EQ(A.is_contiguous(), true);

  A.reshape_({4, 5, 3});
  EXPECT_EQ(A.shape().size(), 3);
  EXPECT_EQ(A.shape()[0], 4);
  EXPECT_EQ(A.shape()[1], 5);
  EXPECT_EQ(A.shape()[2], 3);

  A.reshape_(3, 4, 5);
  EXPECT_EQ(A.shape().size(), 3);
  EXPECT_EQ(A.shape()[0], 3);
  EXPECT_EQ(A.shape()[1], 4);
  EXPECT_EQ(A.shape()[2], 5);

  auto tmp = A.reshape({4, 3, 5});
  EXPECT_EQ(tmp.shape().size(), 3);
  EXPECT_EQ(tmp.shape()[0], 4);
  EXPECT_EQ(tmp.shape()[1], 3);
  EXPECT_EQ(tmp.shape()[2], 5);

  tmp = A.reshape(4, 5, 3);
  EXPECT_EQ(tmp.shape().size(), 3);
  EXPECT_EQ(tmp.shape()[0], 4);
  EXPECT_EQ(tmp.shape()[1], 5);
  EXPECT_EQ(tmp.shape()[2], 3);

  Tensor B({1}, Type.Double, Device.cpu, true);
  EXPECT_EQ(B.shape().size(), 1);
  EXPECT_EQ(B.shape()[0], 1);

  B.reshape_({1, 1});
  EXPECT_EQ(B.shape().size(), 2);
  EXPECT_EQ(B.shape()[0], 1);
  EXPECT_EQ(B.shape()[1], 1);

  B.reshape_({1});
  EXPECT_EQ(B.shape().size(), 1);
  EXPECT_EQ(B.shape()[0], 1);

  B.reshape_(1, 1, 1);
  EXPECT_EQ(B.shape().size(), 3);
  EXPECT_EQ(B.shape()[0], 1);
  EXPECT_EQ(B.shape()[1], 1);
  EXPECT_EQ(B.shape()[2], 1);

  B.reshape_(std::vector<cytnx_int64>{});
  EXPECT_EQ(B.shape().size(), 0);
  B.reshape_(std::vector<cytnx_uint64>{1});
  EXPECT_EQ(B.shape().size(), 1);
  EXPECT_EQ(B.shape()[0], 1);
  EXPECT_FALSE(B.is_scalar());

  EXPECT_THROW(Tensor({0}, Type.Double, Device.cpu, true), std::logic_error);
}

TEST_F(TensorTest, permute) {
  Tensor A({3, 4, 5}, Type.Double, Device.cpu);
  EXPECT_EQ(A.shape().size(), 3);
  EXPECT_EQ(A.shape()[0], 3);
  EXPECT_EQ(A.shape()[1], 4);
  EXPECT_EQ(A.shape()[2], 5);
  EXPECT_EQ(A.is_contiguous(), true);

  A.permute_({2, 1, 0});
  EXPECT_EQ(A.shape().size(), 3);
  EXPECT_EQ(A.shape()[0], 5);
  EXPECT_EQ(A.shape()[1], 4);
  EXPECT_EQ(A.shape()[2], 3);
  EXPECT_EQ(A.is_contiguous(), false);

  A.permute_(2, 1, 0);
  EXPECT_EQ(A.shape().size(), 3);
  EXPECT_EQ(A.shape()[0], 3);
  EXPECT_EQ(A.shape()[1], 4);
  EXPECT_EQ(A.shape()[2], 5);
  EXPECT_EQ(A.is_contiguous(), true);

  auto tmp = A.permute({2, 0, 1});
  EXPECT_EQ(tmp.shape().size(), 3);
  EXPECT_EQ(tmp.shape()[0], 5);
  EXPECT_EQ(tmp.shape()[1], 3);
  EXPECT_EQ(tmp.shape()[2], 4);
  EXPECT_EQ(tmp.is_contiguous(), false);

  tmp = A.permute(2, 0, 1);
  EXPECT_EQ(tmp.shape().size(), 3);
  EXPECT_EQ(tmp.shape()[0], 5);
  EXPECT_EQ(tmp.shape()[1], 3);
  EXPECT_EQ(tmp.shape()[2], 4);
  EXPECT_EQ(tmp.is_contiguous(), false);

  Tensor B({1}, Type.Double, Device.cpu, true);
  EXPECT_EQ(B.shape().size(), 1);
  EXPECT_EQ(B.shape()[0], 1);

  B.permute_({0});
  EXPECT_EQ(B.shape().size(), 1);
  EXPECT_EQ(B.shape()[0], 1);

  B.permute_(0);
  EXPECT_EQ(B.shape().size(), 1);
  EXPECT_EQ(B.shape()[0], 1);

  EXPECT_THROW(Tensor({0}, Type.Double, Device.cpu, true), std::logic_error);
}

TEST_F(TensorTest, get) {
  Tensor tmp = tzero3456(":", ":", ":", ":");
  EXPECT_EQ(tmp.shape().size(), 4);
  EXPECT_EQ(tmp.shape()[0], 3);
  EXPECT_EQ(tmp.shape()[1], 4);
  EXPECT_EQ(tmp.shape()[2], 5);
  EXPECT_EQ(tmp.shape()[3], 6);
  EXPECT_EQ(tmp.is_contiguous(), true);

  tmp = tzero3456(0, ":", ":", ":");
  EXPECT_EQ(tmp.shape().size(), 3);
  EXPECT_EQ(tmp.shape()[0], 4);
  EXPECT_EQ(tmp.shape()[1], 5);
  EXPECT_EQ(tmp.shape()[2], 6);
  EXPECT_EQ(tmp.is_contiguous(), true);

  tmp = tzero3456(0, 0, ":", ":");
  EXPECT_EQ(tmp.shape().size(), 2);
  EXPECT_EQ(tmp.shape()[0], 5);
  EXPECT_EQ(tmp.shape()[1], 6);
  EXPECT_EQ(tmp.is_contiguous(), true);

  tmp = tzero3456(1, 0, 0, ":");
  EXPECT_EQ(tmp.shape().size(), 1);
  EXPECT_EQ(tmp.shape()[0], 6);
  EXPECT_EQ(tmp.is_contiguous(), true);

  tmp = tzero3456(0, 1, 4, 4);
  EXPECT_EQ(tmp.dtype(), Type.ComplexDouble);
  EXPECT_TRUE(tmp.is_scalar());
  EXPECT_EQ(tmp.storage().size(), 1);
  EXPECT_EQ(tmp.item<cytnx_complex128>(), cytnx_complex128(0, 0));
  EXPECT_EQ(tmp.is_contiguous(), true);

  tmp = tarcomplex3456.get({Accessor::all(), Accessor::all(), Accessor::all(), Accessor::all()});
  EXPECT_EQ(tmp.shape().size(), 4);
  EXPECT_EQ(tmp.shape()[0], 3);
  EXPECT_EQ(tmp.shape()[1], 4);
  EXPECT_EQ(tmp.shape()[2], 5);
  EXPECT_EQ(tmp.shape()[3], 6);
  EXPECT_EQ(tmp.is_contiguous(), true);

  tmp = tarcomplex3456(":1", ":", ":", ":");
  EXPECT_EQ(tmp.shape().size(), 4);
  EXPECT_EQ(tmp.shape()[0], 1);
  EXPECT_EQ(tmp.shape()[1], 4);
  EXPECT_EQ(tmp.shape()[2], 5);
  EXPECT_EQ(tmp.shape()[3], 6);
  EXPECT_EQ(tmp.is_contiguous(), true);

  tmp = tarcomplex3456("0:1", ":", ":", ":");
  EXPECT_EQ(tmp.shape().size(), 4);
  EXPECT_EQ(tmp.shape()[0], 1);
  EXPECT_EQ(tmp.shape()[1], 4);
  EXPECT_EQ(tmp.shape()[2], 5);
  EXPECT_EQ(tmp.shape()[3], 6);
  EXPECT_EQ(tmp.is_contiguous(), true);

  tmp = tarcomplex3456("0:2", ":", ":", ":");
  EXPECT_EQ(tmp.shape().size(), 4);
  EXPECT_EQ(tmp.shape()[0], 2);
  EXPECT_EQ(tmp.shape()[1], 4);
  EXPECT_EQ(tmp.shape()[2], 5);
  EXPECT_EQ(tmp.shape()[3], 6);

  tmp = tarcomplex3456("1:2", ":", ":", ":");
  EXPECT_EQ(tmp.shape().size(), 4);
  EXPECT_EQ(tmp.shape()[0], 1);
  EXPECT_EQ(tmp.shape()[1], 4);
  EXPECT_EQ(tmp.shape()[2], 5);
  EXPECT_EQ(tmp.shape()[3], 6);
  TestTools::AreNearlyEqTensor(tmp, tslice1);
}

TEST_F(TensorTest, set) {
  auto tmp = tar3456.clone();
  tar3456(1, 2, 3, 4) = -1999;
  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 4; j++)
      for (size_t k = 0; k < 5; k++)
        for (size_t l = 0; l < 6; l++)
          if (i == 1 && j == 2 && k == 3 && l == 4) {
            EXPECT_EQ(tar3456(i, j, k, l).item().real(), -1999);
          } else {
            EXPECT_EQ(tar3456(i, j, k, l).item().real(), tmp(i, j, k, l).item().real());
          }
}

TEST_F(TensorTest, RankZeroScalarAccessAndSet) {
  Tensor scalar(std::vector<cytnx_uint64>{}, Type.Double);
  scalar.item<double>() = 3.25;
  EXPECT_DOUBLE_EQ(scalar.item<double>(), 3.25);
  EXPECT_DOUBLE_EQ(scalar.at<double>({}), 3.25);

  scalar.at<double>({}) = 4.5;
  EXPECT_DOUBLE_EQ(scalar.item<double>(), 4.5);

  Tensor selected = scalar.get(std::vector<Accessor>{});
  EXPECT_EQ(selected.shape().size(), 0);
  EXPECT_TRUE(selected.is_scalar());
  EXPECT_DOUBLE_EQ(selected.item<double>(), 4.5);

  Tensor replacement(std::vector<cytnx_uint64>{}, Type.Double);
  replacement.item<double>() = -2.0;
  scalar.set(std::vector<Accessor>{}, replacement);
  EXPECT_DOUBLE_EQ(scalar.item<double>(), -2.0);

  scalar.set(std::vector<Accessor>{}, 7.0);
  EXPECT_DOUBLE_EQ(scalar.item<double>(), 7.0);
}

TEST_F(TensorTest, RankZeroBroadcastArithmetic) {
  Tensor scalar(std::vector<cytnx_uint64>{}, Type.Double);
  scalar.item<double>() = 2.0;
  Tensor vec = arange(3).astype(Type.Double);

  Tensor out = scalar + vec;
  EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(out.at<double>({0}), 2.0);
  EXPECT_DOUBLE_EQ(out.at<double>({1}), 3.0);
  EXPECT_DOUBLE_EQ(out.at<double>({2}), 4.0);

  out = vec + scalar;
  EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(out.at<double>({0}), 2.0);
  EXPECT_DOUBLE_EQ(out.at<double>({1}), 3.0);
  EXPECT_DOUBLE_EQ(out.at<double>({2}), 4.0);

  out = scalar - vec;
  EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(out.at<double>({0}), 2.0);
  EXPECT_DOUBLE_EQ(out.at<double>({1}), 1.0);
  EXPECT_DOUBLE_EQ(out.at<double>({2}), 0.0);

  out = vec - scalar;
  EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(out.at<double>({0}), -2.0);
  EXPECT_DOUBLE_EQ(out.at<double>({1}), -1.0);
  EXPECT_DOUBLE_EQ(out.at<double>({2}), 0.0);

  Tensor scalar2(std::vector<cytnx_uint64>{}, Type.Double);
  scalar2.item<double>() = 5.0;
  out = scalar * scalar2;
  EXPECT_EQ(out.shape().size(), 0);
  EXPECT_DOUBLE_EQ(out.item<double>(), 10.0);

  out = scalar * vec;
  EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(out.at<double>({0}), 0.0);
  EXPECT_DOUBLE_EQ(out.at<double>({1}), 2.0);
  EXPECT_DOUBLE_EQ(out.at<double>({2}), 4.0);

  out = vec * scalar;
  EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(out.at<double>({0}), 0.0);
  EXPECT_DOUBLE_EQ(out.at<double>({1}), 2.0);
  EXPECT_DOUBLE_EQ(out.at<double>({2}), 4.0);

  Tensor denom = arange(1, 4, 1, Type.Double);
  out = scalar / denom;
  EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(out.at<double>({0}), 2.0);
  EXPECT_DOUBLE_EQ(out.at<double>({1}), 1.0);
  EXPECT_DOUBLE_EQ(out.at<double>({2}), 2.0 / 3.0);

  out = denom / scalar;
  EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(out.at<double>({0}), 0.5);
  EXPECT_DOUBLE_EQ(out.at<double>({1}), 1.0);
  EXPECT_DOUBLE_EQ(out.at<double>({2}), 1.5);

  Tensor mod_scalar(std::vector<cytnx_uint64>{}, Type.Int64);
  mod_scalar.item<cytnx_int64>() = 5;
  Tensor mod_vec = arange(2, 5, 1, Type.Int64);

  out = linalg::Mod(mod_scalar, mod_vec);
  EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_EQ(out.at<cytnx_int64>({0}), 1);
  EXPECT_EQ(out.at<cytnx_int64>({1}), 2);
  EXPECT_EQ(out.at<cytnx_int64>({2}), 1);

  out = linalg::Mod(mod_vec, mod_scalar);
  EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_EQ(out.at<cytnx_int64>({0}), 2);
  EXPECT_EQ(out.at<cytnx_int64>({1}), 3);
  EXPECT_EQ(out.at<cytnx_int64>({2}), 4);

  vec += scalar;
  EXPECT_EQ(vec.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(vec.at<double>({0}), 2.0);
  EXPECT_DOUBLE_EQ(vec.at<double>({1}), 3.0);
  EXPECT_DOUBLE_EQ(vec.at<double>({2}), 4.0);

  EXPECT_THROW(scalar += vec, std::logic_error);

  Tensor legacy_shape_one({1}, Type.Double);
  legacy_shape_one.at<double>({0}) = 2.0;
  EXPECT_FALSE(legacy_shape_one.is_scalar());
  EXPECT_DOUBLE_EQ(legacy_shape_one.item<double>(), 2.0);
  Tensor legacy_shape_one_one({1, 1}, Type.Double);
  legacy_shape_one_one.at<double>({0, 0}) = 3.0;
  EXPECT_FALSE(legacy_shape_one_one.is_scalar());
  EXPECT_DOUBLE_EQ(legacy_shape_one_one.item<double>(), 3.0);
  EXPECT_THROW((void)(legacy_shape_one + vec), std::logic_error);
  EXPECT_THROW((void)(vec + legacy_shape_one), std::logic_error);
  EXPECT_THROW(vec += legacy_shape_one, std::logic_error);
}

TEST_F(TensorTest, RankZeroVectordotIsScalar) {
  Tensor vec = arange(3).astype(Type.Double);

  Tensor dot = linalg::Vectordot(vec, vec, false);
  EXPECT_TRUE(dot.is_scalar());
  EXPECT_EQ(dot.shape().size(), 0);
  EXPECT_DOUBLE_EQ(dot.item<double>(), 5.0);

  Tensor scaled = dot * vec;
  EXPECT_EQ(scaled.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(scaled.at<double>({0}), 0.0);
  EXPECT_DOUBLE_EQ(scaled.at<double>({1}), 5.0);
  EXPECT_DOUBLE_EQ(scaled.at<double>({2}), 10.0);
}

TEST_F(TensorTest, RankZeroReductionsAreScalarAndBroadcast) {
  Tensor vec = arange(1, 4, 1, Type.Double);

  Tensor sum = linalg::Sum(vec);
  EXPECT_TRUE(sum.is_scalar());
  EXPECT_DOUBLE_EQ(sum.item<double>(), 6.0);

  Tensor max = linalg::Max(vec);
  EXPECT_TRUE(max.is_scalar());
  EXPECT_DOUBLE_EQ(max.item<double>(), 3.0);

  Tensor min = linalg::Min(vec);
  EXPECT_TRUE(min.is_scalar());
  EXPECT_DOUBLE_EQ(min.item<double>(), 1.0);

  Tensor shifted = sum + vec;
  EXPECT_EQ(shifted.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(shifted.at<double>({0}), 7.0);
  EXPECT_DOUBLE_EQ(shifted.at<double>({1}), 8.0);
  EXPECT_DOUBLE_EQ(shifted.at<double>({2}), 9.0);
}

TEST_F(TensorTest, RankZeroBroadcastComparison) {
  Tensor scalar(std::vector<cytnx_uint64>{}, Type.Double);
  scalar.item<double>() = 2.0;
  Tensor vec = arange(3).astype(Type.Double);

  Tensor out = linalg::Cpr(scalar, vec);
  EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_FALSE(out.at<cytnx_bool>({0}));
  EXPECT_FALSE(out.at<cytnx_bool>({1}));
  EXPECT_TRUE(out.at<cytnx_bool>({2}));
}

TEST_F(TensorTest, RankZeroDiagThrowsControlledError) {
  Tensor scalar(std::vector<cytnx_uint64>{}, Type.Double);
  scalar.item<double>() = 2.0;
  EXPECT_THROW(linalg::Diag(scalar), std::logic_error);
}

TEST_F(TensorTest, RankZeroSaveLoad) {
  Tensor scalar(std::vector<cytnx_uint64>{}, Type.Double);
  scalar.item<double>() = 8.25;

  const auto path = std::filesystem::temp_directory_path() / "cytnx_rank_zero_tensor.cytn";
  scalar.Save(path.string());

  {
    std::ifstream header(path, std::ios::binary);
    ASSERT_TRUE(header.is_open());
    unsigned int magic = 0;
    unsigned int version = 0;
    header.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    header.read(reinterpret_cast<char *>(&version), sizeof(version));
    EXPECT_EQ(magic, 889);
    EXPECT_EQ(version, 1);
  }

  Tensor loaded = Tensor::Load(path.string());
  EXPECT_EQ(loaded.shape().size(), 0);
  EXPECT_EQ(loaded.rank(), 0);
  EXPECT_EQ(loaded.dtype(), Type.Double);
  EXPECT_DOUBLE_EQ(loaded.item<double>(), 8.25);
  std::filesystem::remove(path);

  Tensor vector_one({1}, Type.Double);
  vector_one.at<double>({0}) = 3.5;
  const auto vector_path =
    std::filesystem::temp_directory_path() / "cytnx_rank_one_size_one_tensor.cytn";
  vector_one.Save(vector_path.string());

  Tensor loaded_vector_one = Tensor::Load(vector_path.string());
  EXPECT_EQ(loaded_vector_one.shape(), (std::vector<cytnx_uint64>{1}));
  EXPECT_FALSE(loaded_vector_one.is_scalar());
  EXPECT_DOUBLE_EQ(loaded_vector_one.at<double>({0}), 3.5);
  std::filesystem::remove(vector_path);
}

TEST_F(TensorTest, identity) {
  Tensor tn = cytnx::identity(2, Type.Double, Device.cpu);
  EXPECT_EQ(tn.shape().size(), 2);
  EXPECT_EQ(tn.shape()[0], 2);
  EXPECT_EQ(tn.shape()[1], 2);
  EXPECT_EQ(tn.is_contiguous(), true);
  EXPECT_EQ(tn.dtype(), Type.Double);
  EXPECT_EQ(tn.device(), Device.cpu);
  EXPECT_DOUBLE_EQ((double)tn(0, 0).item().real(), 1);
  EXPECT_DOUBLE_EQ((double)tn(1, 1).item().real(), 1);
  EXPECT_DOUBLE_EQ((double)tn(0, 1).item().real(), 0);
  EXPECT_DOUBLE_EQ((double)tn(1, 0).item().real(), 0);

  tn = cytnx::identity(3, Type.Double, Device.cpu);
  EXPECT_EQ(tn.shape().size(), 2);
  EXPECT_EQ(tn.shape()[0], 3);
  EXPECT_EQ(tn.shape()[1], 3);
  EXPECT_EQ(tn.is_contiguous(), true);
  EXPECT_EQ(tn.dtype(), Type.Double);
  EXPECT_EQ(tn.device(), Device.cpu);
  EXPECT_DOUBLE_EQ(tn.at<double>({0, 0}), 1);
  EXPECT_DOUBLE_EQ(tn.at<double>({1, 1}), 1);
  EXPECT_DOUBLE_EQ(tn.at<double>({0, 1}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({1, 0}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({2, 0}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({2, 1}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({0, 2}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({1, 2}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({2, 2}), 1);
}

TEST_F(TensorTest, eye) {
  Tensor tn = cytnx::eye(2, Type.Double, Device.cpu);
  EXPECT_EQ(tn.shape().size(), 2);
  EXPECT_EQ(tn.shape()[0], 2);
  EXPECT_EQ(tn.shape()[1], 2);
  EXPECT_EQ(tn.is_contiguous(), true);
  EXPECT_EQ(tn.dtype(), Type.Double);
  EXPECT_EQ(tn.device(), Device.cpu);
  EXPECT_DOUBLE_EQ((double)tn(0, 0).item().real(), 1);
  EXPECT_DOUBLE_EQ((double)tn(1, 1).item().real(), 1);
  EXPECT_DOUBLE_EQ((double)tn(0, 1).item().real(), 0);
  EXPECT_DOUBLE_EQ((double)tn(1, 0).item().real(), 0);

  tn = cytnx::eye(3, Type.Double, Device.cpu);
  EXPECT_EQ(tn.shape().size(), 2);
  EXPECT_EQ(tn.shape()[0], 3);
  EXPECT_EQ(tn.shape()[1], 3);
  EXPECT_EQ(tn.is_contiguous(), true);
  EXPECT_EQ(tn.dtype(), Type.Double);
  EXPECT_EQ(tn.device(), Device.cpu);
  EXPECT_DOUBLE_EQ(tn.at<double>({0, 0}), 1);
  EXPECT_DOUBLE_EQ(tn.at<double>({1, 1}), 1);
  EXPECT_DOUBLE_EQ(tn.at<double>({0, 1}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({1, 0}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({2, 0}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({2, 1}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({0, 2}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({1, 2}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({2, 2}), 1);
}
// TEST_F(TensorTest, approx_eq) {
//   cytnx::User_debug = true;
//   EXPECT_TRUE(tar3456.approx_eq(tar3456));
//   EXPECT_FALSE(tar345.approx_eq(tar345.permute(1, 0, 2)));
//   EXPECT_FALSE(tar3456.approx_eq(tarcomplex3456));
//   EXPECT_TRUE(tone3456.approx_eq(tone3456.astype(Type.ComplexFloat), 1e-5));
// }

TEST(Tensor, ItemDtypeMismatchThrows) {
  Tensor t = zeros({1}, Type.Float);
  EXPECT_THROW(t.item<double>(), std::logic_error);
  EXPECT_NO_THROW(t.item<float>());
}

TEST(Tensor, ReshapeRejectsMultipleUnknownDims) {
  Tensor t = zeros({12}, Type.Double);
  EXPECT_THROW(t.reshape({-1, -1}), std::logic_error);
  EXPECT_THROW(t.reshape_({-1, -1}), std::logic_error);
  EXPECT_THROW(t.reshape({-2, 6}), std::logic_error);
  // a single -1 must keep working
  Tensor r = t.reshape({3, -1});
  EXPECT_EQ(r.shape(), (std::vector<cytnx_uint64>{3, 4}));
}

TEST(Tensor, ReshapeRejectsZeroDimWithUnknownDim) {
  Tensor t = zeros({12}, Type.Double);
  // new_N == 0 previously fed a modulo/division by zero (UB / SIGFPE on x86)
  EXPECT_THROW(t.reshape({0, -1}), std::logic_error);
  EXPECT_THROW(t.reshape_({0, -1}), std::logic_error);
  EXPECT_THROW(t.reshape_({-1, 0}), std::logic_error);
}

TEST(Tensor, FailedReshapeLeavesShapeUnchanged) {
  Tensor t = zeros({12}, Type.Double);
  EXPECT_THROW(t.reshape_({7, 2}), std::logic_error);
  EXPECT_EQ(t.shape(), (std::vector<cytnx_uint64>{12}));
  EXPECT_THROW(t.reshape_({5, -1}), std::logic_error);
  EXPECT_EQ(t.shape(), (std::vector<cytnx_uint64>{12}));
  EXPECT_THROW(t.reshape_({0, -1}), std::logic_error);
  EXPECT_EQ(t.shape(), (std::vector<cytnx_uint64>{12}));
}

// NOTE: `Tensor b = a;` makes `b` an alias of the *same* Tensor_impl as `a`
// (Tensor's copy ctor just copies the intrusive_ptr), so `is(a.storage(),
// b.storage())` would trivially read the same Storage field twice and could
// never observe the #906 detach bug. To exercise a genuine "two independent
// Tensor handles sharing one Storage" scenario (e.g. what a view/slice would
// produce), we build `b` via Tensor::from_storage(a.storage()), which creates
// a brand-new Tensor_impl whose _storage is a copy of the Storage handle
// (same underlying Storage_base, distinct Tensor_impl).
TEST(Tensor, ScalarInplaceAddKeepsStorageSharing) {
  Tensor a = zeros({4}, Type.Double);
  Tensor b = Tensor::from_storage(a.storage());  // distinct Tensor_impl, shared Storage
  ASSERT_TRUE(is(a.storage(), b.storage()));
  a += 1.0;
  EXPECT_TRUE(is(a.storage(), b.storage()));
  EXPECT_DOUBLE_EQ(b.storage().at<double>(0), 1.0);
}

TEST(Tensor, ScalarInplaceOpsPreserveDtype) {
  Tensor a = ones({2}, Type.Float);
  a += 1.0;  // double scalar must not promote the tensor
  a -= 0.5;
  a *= 2.0;
  a /= 3.0;
  EXPECT_EQ(a.dtype(), Type.Float);
  EXPECT_FLOAT_EQ(a.storage().at<float>(0), 1.0f);
}

TEST(Tensor, ScalarInplaceRealPlusComplexThrows) {
  Tensor a = zeros({2}, Type.Double);
  EXPECT_THROW(a += cytnx_complex128(0, 1), std::logic_error);
  EXPECT_THROW(a -= cytnx_complex128(0, 1), std::logic_error);
  EXPECT_THROW(a *= cytnx_complex128(0, 1), std::logic_error);
  EXPECT_THROW(a /= cytnx_complex128(0, 1), std::logic_error);
}

TEST(Tensor, ScalarInplaceIntTensorTruncatesFractionalScalar) {
  Tensor a = ones({2}, Type.Int64);
  a += 2.7;  // was: promoted to Double (3.7); now: stays Int64, truncates
  EXPECT_EQ(a.dtype(), Type.Int64);
  EXPECT_EQ(a.storage().at<cytnx_int64>(0), 3);
}

// Mirrors the actual #906 report through public API: permute() produces a
// distinct Tensor_impl sharing the same Storage (Tensor_impl::permute does
// `out->_storage = this->_storage`) flagged non-contiguous, so this also
// exercises the non-contiguous scalar broadcast path of iAdd.
TEST(Tensor, ScalarInplaceOnPermutedViewMutatesSharedStorage) {
  Tensor a = zeros({2, 3}, Type.Double);
  Tensor v = a.permute({1, 0});  // distinct impl, shared storage, non-contiguous
  ASSERT_FALSE(v.is_contiguous());
  ASSERT_TRUE(is(a.storage(), v.storage()));
  v += 1.0;
  EXPECT_TRUE(is(a.storage(), v.storage()));
  EXPECT_DOUBLE_EQ(a.storage().at<double>(0), 1.0);
}

TEST(Tensor, ScalarInplaceSubMulDivKeepStorageSharing) {
  Tensor a = ones({3}, Type.Double);
  Tensor b = Tensor::from_storage(a.storage());
  a -= 0.5;
  a *= 4.0;
  a /= 2.0;
  EXPECT_TRUE(is(a.storage(), b.storage()));
  EXPECT_DOUBLE_EQ(b.storage().at<double>(2), 1.0);
}

TEST(Tensor, CytnxScalarInplaceAddKeepsStorageSharing) {
  Tensor a = zeros({2}, Type.Double);
  Tensor b = Tensor::from_storage(a.storage());
  a += Scalar(2.5);
  EXPECT_TRUE(is(a.storage(), b.storage()));
  EXPECT_DOUBLE_EQ(b.storage().at<double>(1), 2.5);
}

TEST(Tensor, ScalarInplaceRealTimesComplexErrorNamesOperator) {
  Tensor a = zeros({2}, Type.Double);
  try {
    a *= cytnx_complex128(0, 1);
    FAIL() << "expected real *= complex to throw";
  } catch (const std::logic_error &e) {
    EXPECT_NE(std::string(e.what()).find("*="), std::string::npos) << e.what();
  }
  try {
    a /= cytnx_complex128(0, 1);
    FAIL() << "expected real /= complex to throw";
  } catch (const std::logic_error &e) {
    EXPECT_NE(std::string(e.what()).find("/="), std::string::npos) << e.what();
  }
}
