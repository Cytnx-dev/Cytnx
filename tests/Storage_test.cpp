#include "Storage_test.h"
#include "test_tools.h"
#include <vector>

TEST_F(StorageTest, dtype_str) {
  std::vector<cytnx_complex128> vcd = {cytnx_complex128(1, 2), cytnx_complex128(3, 4),
                                       cytnx_complex128(5, 6)};

  Storage sd = Storage::from_vector(vcd);
  EXPECT_EQ(sd.dtype_str(), Type.getname(Type.ComplexDouble));
}

TEST_F(StorageTest, device_str) {
  std::vector<cytnx_complex128> vcd = {cytnx_complex128(1, 2), cytnx_complex128(3, 4),
                                       cytnx_complex128(5, 6)};

  Storage sd = Storage::from_vector(vcd);
  EXPECT_EQ(sd.device_str(), Device.getname(Device.cpu));
}

TEST_F(StorageTest, Get_real_cd) {
  std::vector<cytnx_complex128> vcd = {cytnx_complex128(1, 2), cytnx_complex128(3, 4),
                                       cytnx_complex128(5, 6)};

  Storage sd = Storage::from_vector(vcd);

  Storage real_sd = sd.real();

  EXPECT_EQ(real_sd.at<double>(0), double(1));
  EXPECT_EQ(real_sd.at<double>(1), double(3));
  EXPECT_EQ(real_sd.at<double>(2), double(5));
}

TEST_F(StorageTest, Get_imag_cd) {
  std::vector<cytnx_complex128> vcd = {cytnx_complex128(1, 2), cytnx_complex128(3, 4),
                                       cytnx_complex128(5, 6)};

  Storage sd = Storage::from_vector(vcd);

  Storage im_sd = sd.imag();

  EXPECT_EQ(im_sd.at<double>(0), double(2));
  EXPECT_EQ(im_sd.at<double>(1), double(4));
  EXPECT_EQ(im_sd.at<double>(2), double(6));
}

TEST_F(StorageTest, Get_real_cf) {
  std::vector<cytnx_complex64> vcf = {cytnx_complex64(1, 2), cytnx_complex64(3, 4),
                                      cytnx_complex64(5, 6)};

  Storage sd = Storage::from_vector(vcf);

  Storage real_sd = sd.real();

  EXPECT_EQ(real_sd.at<float>(0), float(1));
  EXPECT_EQ(real_sd.at<float>(1), float(3));
  EXPECT_EQ(real_sd.at<float>(2), float(5));
}

TEST_F(StorageTest, Get_imag_cf) {
  std::vector<cytnx_complex64> vcf = {cytnx_complex64(1, 2), cytnx_complex64(3, 4),
                                      cytnx_complex64(5, 6)};

  Storage sd = Storage::from_vector(vcf);

  Storage im_sd = sd.imag();

  EXPECT_EQ(im_sd.at<float>(0), float(2));
  EXPECT_EQ(im_sd.at<float>(1), float(4));
  EXPECT_EQ(im_sd.at<float>(2), float(6));
}

// test fromvector:

TEST_F(StorageTest, from_vec_cd) {
  auto e1 = cytnx_complex128(1, 2);
  auto e2 = cytnx_complex128(3, 4);
  std::vector<cytnx_complex128> vcd = {e1, e2};
  Storage sd = Storage::from_vector(vcd);

  EXPECT_EQ(sd.dtype(), Type.ComplexDouble);
  EXPECT_EQ(sd.at<cytnx_complex128>(0), e1);
  EXPECT_EQ(sd.at<cytnx_complex128>(1), e2);
}

TEST_F(StorageTest, from_vec_cf) {
  auto e1 = cytnx_complex64(1, 2);
  auto e2 = cytnx_complex64(3, 4);
  std::vector<cytnx_complex64> vcd = {e1, e2};
  Storage sd = Storage::from_vector(vcd);

  EXPECT_EQ(sd.dtype(), Type.ComplexFloat);
  EXPECT_EQ(sd.at<cytnx_complex64>(0), e1);
  EXPECT_EQ(sd.at<cytnx_complex64>(1), e2);
}

// create suite for all real types (exclude bool)
using vector_typelist = testing::Types<cytnx_int64, cytnx_uint64, cytnx_int32, cytnx_uint32,
                                       cytnx_double, cytnx_float, cytnx_uint16, cytnx_int64>;
template <class>
struct vector_suite : testing::Test {};
TYPED_TEST_SUITE(vector_suite, vector_typelist);

// generic testing for all type:
TYPED_TEST(vector_suite, from_vec_real) {
  auto e1 = TypeParam(2);
  auto e2 = TypeParam(7);

  std::vector<TypeParam> v = {e1, e2};
  Storage sd = Storage::from_vector(v);

  EXPECT_EQ(sd.dtype(), Type.cy_typeid(e1));
  EXPECT_EQ(sd.at<TypeParam>(0), e1);
  EXPECT_EQ(sd.at<TypeParam>(1), e2);
}

TYPED_TEST(vector_suite, storage_cpu_to_cpu) {
  auto e1 = TypeParam(2);
  auto e2 = TypeParam(7);

  std::vector<TypeParam> v = {e1, e2};
  Storage sd = Storage::from_vector(v);

  EXPECT_EQ(sd.dtype(), Type.cy_typeid(e1));

  Storage tar_sd = sd.to(Device.cpu);
  EXPECT_EQ(tar_sd.device(), Device.cpu);
  EXPECT_EQ(is(tar_sd, sd), true);
}

TYPED_TEST(vector_suite, storage_fill_d) {
  auto e1 = TypeParam(2);
  auto e2 = TypeParam(7);

  std::vector<TypeParam> v = {e1, e2};
  Storage sd = Storage::from_vector(v);

  auto fille = double(10);
  sd.fill(fille);

  EXPECT_EQ(sd.at<TypeParam>(0), TypeParam(fille));
  EXPECT_EQ(sd.at<TypeParam>(1), TypeParam(fille));
}

TYPED_TEST(vector_suite, storage_fill_f) {
  auto e1 = TypeParam(2);
  auto e2 = TypeParam(7);

  std::vector<TypeParam> v = {e1, e2};
  Storage sd = Storage::from_vector(v);

  auto fille = float(10);
  sd.fill(fille);
  cout << sd << endl;

  EXPECT_EQ(sd.at<TypeParam>(0), TypeParam(fille));
  EXPECT_EQ(sd.at<TypeParam>(1), TypeParam(fille));
}

TYPED_TEST(vector_suite, storage_fill_i64) {
  auto e1 = TypeParam(2);
  auto e2 = TypeParam(7);

  std::vector<TypeParam> v = {e1, e2};
  Storage sd = Storage::from_vector(v);

  auto fille = cytnx_int64(10);
  sd.fill(fille);

  EXPECT_EQ(sd.at<TypeParam>(0), TypeParam(fille));
  EXPECT_EQ(sd.at<TypeParam>(1), TypeParam(fille));
}

TYPED_TEST(vector_suite, storage_fill_u64) {
  auto e1 = TypeParam(2);
  auto e2 = TypeParam(7);

  std::vector<TypeParam> v = {e1, e2};
  Storage sd = Storage::from_vector(v);

  auto fille = cytnx_uint64(10);
  sd.fill(fille);

  EXPECT_EQ(sd.at<TypeParam>(0), TypeParam(fille));
  EXPECT_EQ(sd.at<TypeParam>(1), TypeParam(fille));
}

TYPED_TEST(vector_suite, storage_fill_i32) {
  auto e1 = TypeParam(2);
  auto e2 = TypeParam(7);

  std::vector<TypeParam> v = {e1, e2};
  Storage sd = Storage::from_vector(v);

  auto fille = cytnx_int32(10);
  sd.fill(fille);

  EXPECT_EQ(sd.at<TypeParam>(0), TypeParam(fille));
  EXPECT_EQ(sd.at<TypeParam>(1), TypeParam(fille));
}

TYPED_TEST(vector_suite, storage_fill_u32) {
  auto e1 = TypeParam(2);
  auto e2 = TypeParam(7);

  std::vector<TypeParam> v = {e1, e2};
  Storage sd = Storage::from_vector(v);

  auto fille = cytnx_uint32(10);
  sd.fill(fille);

  EXPECT_EQ(sd.at<TypeParam>(0), TypeParam(fille));
  EXPECT_EQ(sd.at<TypeParam>(1), TypeParam(fille));
}

TYPED_TEST(vector_suite, storage_fill_i16) {
  auto e1 = TypeParam(2);
  auto e2 = TypeParam(7);

  std::vector<TypeParam> v = {e1, e2};
  Storage sd = Storage::from_vector(v);

  auto fille = cytnx_int16(10);
  sd.fill(fille);

  EXPECT_EQ(sd.at<TypeParam>(0), TypeParam(fille));
  EXPECT_EQ(sd.at<TypeParam>(1), TypeParam(fille));
}

TYPED_TEST(vector_suite, storage_fill_u16) {
  auto e1 = TypeParam(2);
  auto e2 = TypeParam(7);

  std::vector<TypeParam> v = {e1, e2};
  Storage sd = Storage::from_vector(v);

  auto fille = cytnx_uint16(10);
  sd.fill(fille);

  EXPECT_EQ(sd.at<TypeParam>(0), TypeParam(fille));
  EXPECT_EQ(sd.at<TypeParam>(1), TypeParam(fille));
}

TYPED_TEST(vector_suite, storage_fill_b) {
  auto e1 = TypeParam(2);
  auto e2 = TypeParam(7);

  std::vector<TypeParam> v = {e1, e2};
  Storage sd = Storage::from_vector(v);

  auto fille = true;
  sd.fill(fille);

  EXPECT_EQ(sd.at<TypeParam>(0), TypeParam(fille));
  EXPECT_EQ(sd.at<TypeParam>(1), TypeParam(fille));
}
