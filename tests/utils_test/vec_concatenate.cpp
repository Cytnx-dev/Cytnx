#include "vec_test.h"

TEST_F(VecTest, vec_concatenate) {
  EXPECT_EQ(std::vector<cytnx::cytnx_uint64>(), cytnx::vec_concatenate(ui64v, ui64v));
  EXPECT_EQ(std::vector<cytnx::cytnx_uint32>(), cytnx::vec_concatenate(ui32v, ui32v));
  EXPECT_EQ(std::vector<cytnx::cytnx_uint16>(), cytnx::vec_concatenate(ui16v, ui16v));
  EXPECT_EQ(std::vector<cytnx::cytnx_int64>(), cytnx::vec_concatenate(i64v, i64v));
  EXPECT_EQ(std::vector<cytnx::cytnx_int32>(), cytnx::vec_concatenate(i32v, i32v));
  EXPECT_EQ(std::vector<cytnx::cytnx_int16>(), cytnx::vec_concatenate(i16v, i16v));
  EXPECT_EQ(std::vector<cytnx::cytnx_double>(), cytnx::vec_concatenate(doublev, doublev));
  EXPECT_EQ(std::vector<cytnx::cytnx_float>(), cytnx::vec_concatenate(floatv, floatv));
  EXPECT_EQ(std::vector<cytnx::cytnx_complex128>(), cytnx::vec_concatenate(c128v, c128v));
  EXPECT_EQ(std::vector<cytnx::cytnx_complex64>(), cytnx::vec_concatenate(c64v, c64v));
  EXPECT_EQ(std::vector<cytnx::cytnx_bool>(), cytnx::vec_concatenate(boolv, boolv));

  EXPECT_EQ(std::vector<cytnx::cytnx_uint64>(), cytnx::vec_concatenate(ui64v, ui64v2));
  EXPECT_EQ(std::vector<cytnx::cytnx_uint32>(), cytnx::vec_concatenate(ui32v, ui32v2));
  EXPECT_EQ(std::vector<cytnx::cytnx_uint16>(), cytnx::vec_concatenate(ui16v, ui16v2));
  EXPECT_EQ(std::vector<cytnx::cytnx_int64>(), cytnx::vec_concatenate(i64v, i64v2));
  EXPECT_EQ(std::vector<cytnx::cytnx_int32>(), cytnx::vec_concatenate(i32v, i32v2));
  EXPECT_EQ(std::vector<cytnx::cytnx_int16>(), cytnx::vec_concatenate(i16v, i16v2));
  EXPECT_EQ(std::vector<cytnx::cytnx_double>(), cytnx::vec_concatenate(doublev, doublev2));
  EXPECT_EQ(std::vector<cytnx::cytnx_float>(), cytnx::vec_concatenate(floatv, floatv2));
  EXPECT_EQ(std::vector<cytnx::cytnx_complex128>(), cytnx::vec_concatenate(c128v, c128v2));
  EXPECT_EQ(std::vector<cytnx::cytnx_complex64>(), cytnx::vec_concatenate(c64v, c64v2));
  EXPECT_EQ(std::vector<cytnx::cytnx_bool>(), cytnx::vec_concatenate(boolv, boolv2));

  cytnx::vec_concatenate_(ui64v3, ui64v, ui64v);
  cytnx::vec_concatenate_(ui32v3, ui32v, ui32v);
  cytnx::vec_concatenate_(ui16v3, ui16v, ui16v);
  cytnx::vec_concatenate_(i64v3, i64v, i64v);
  cytnx::vec_concatenate_(i32v3, i32v, i32v);
  cytnx::vec_concatenate_(i16v3, i16v, i16v);
  cytnx::vec_concatenate_(doublev3, doublev, doublev);
  cytnx::vec_concatenate_(floatv3, floatv, floatv);
  cytnx::vec_concatenate_(c128v3, c128v, c128v);
  cytnx::vec_concatenate_(c64v3, c64v, c64v);
  cytnx::vec_concatenate_(boolv3, boolv, boolv);
  EXPECT_EQ(std::vector<cytnx::cytnx_uint64>(), ui64v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_uint32>(), ui32v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_uint16>(), ui16v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_int64>(), i64v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_int32>(), i32v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_int16>(), i16v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_double>(), doublev3);
  EXPECT_EQ(std::vector<cytnx::cytnx_float>(), floatv3);
  EXPECT_EQ(std::vector<cytnx::cytnx_complex128>(), c128v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_complex64>(), c64v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_bool>(), boolv3);

  cytnx::vec_concatenate_(ui64v3, ui64v, ui64v2);
  cytnx::vec_concatenate_(ui32v3, ui32v, ui32v2);
  cytnx::vec_concatenate_(ui16v3, ui16v, ui16v2);
  cytnx::vec_concatenate_(i64v3, i64v, i64v2);
  cytnx::vec_concatenate_(i32v3, i32v, i32v2);
  cytnx::vec_concatenate_(i16v3, i16v, i16v2);
  cytnx::vec_concatenate_(doublev3, doublev, doublev2);
  cytnx::vec_concatenate_(floatv3, floatv, floatv2);
  cytnx::vec_concatenate_(c128v3, c128v, c128v2);
  cytnx::vec_concatenate_(c64v3, c64v, c64v2);
  cytnx::vec_concatenate_(boolv3, boolv, boolv2);
  EXPECT_EQ(std::vector<cytnx::cytnx_uint64>(), ui64v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_uint32>(), ui32v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_uint16>(), ui16v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_int64>(), i64v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_int32>(), i32v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_int16>(), i16v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_double>(), doublev3);
  EXPECT_EQ(std::vector<cytnx::cytnx_float>(), floatv3);
  EXPECT_EQ(std::vector<cytnx::cytnx_complex128>(), c128v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_complex64>(), c64v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_bool>(), boolv3);

  ui64v.insert(ui64v.end(), {1, 15, 7});
  ui64v2.insert(ui64v2.end(), {1, 15});
  ui32v.insert(ui32v.end(), {1, 15, 7});
  ui32v2.insert(ui32v2.end(), {1, 15});
  ui16v.insert(ui16v.end(), {1, 15, 7});
  ui16v2.insert(ui16v2.end(), {1, 15});
  i64v.insert(i64v.end(), {1, 15, 7});
  i64v2.insert(i64v2.end(), {1, 15});
  i32v.insert(i32v.end(), {1, 15, 7});
  i32v2.insert(i32v2.end(), {1, 15});
  i16v.insert(i16v.end(), {1, 15, 7});
  i16v2.insert(i16v2.end(), {1, 15});
  doublev.insert(doublev.end(), {1, 15, 7});
  doublev2.insert(doublev2.end(), {1, 15});
  floatv.insert(floatv.end(), {1, 15, 7});
  floatv2.insert(floatv2.end(), {1, 15});
  c128v.insert(c128v.end(), {1, 15, 7});
  c128v2.insert(c128v2.end(), {1, 15});
  c64v.insert(c64v.end(), {1, 15, 7});
  c64v2.insert(c64v2.end(), {1, 15});
  boolv.insert(boolv.end(), {1, 0});
  boolv2.insert(boolv2.end(), {1, 0, 0});

  EXPECT_EQ(std::vector<cytnx::cytnx_uint64>({1, 15, 7, 1, 15}),
            cytnx::vec_concatenate(ui64v, ui64v2));
  EXPECT_EQ(std::vector<cytnx::cytnx_uint32>({1, 15, 7, 1, 15}),
            cytnx::vec_concatenate(ui32v, ui32v2));
  EXPECT_EQ(std::vector<cytnx::cytnx_uint16>({1, 15, 7, 1, 15}),
            cytnx::vec_concatenate(ui16v, ui16v2));
  EXPECT_EQ(std::vector<cytnx::cytnx_int64>({1, 15, 7, 1, 15}),
            cytnx::vec_concatenate(i64v, i64v2));
  EXPECT_EQ(std::vector<cytnx::cytnx_int32>({1, 15, 7, 1, 15}),
            cytnx::vec_concatenate(i32v, i32v2));
  EXPECT_EQ(std::vector<cytnx::cytnx_int16>({1, 15, 7, 1, 15}),
            cytnx::vec_concatenate(i16v, i16v2));
  EXPECT_EQ(std::vector<cytnx::cytnx_double>({1, 15, 7, 1, 15}),
            cytnx::vec_concatenate(doublev, doublev2));
  EXPECT_EQ(std::vector<cytnx::cytnx_float>({1, 15, 7, 1, 15}),
            cytnx::vec_concatenate(floatv, floatv2));
  EXPECT_EQ(std::vector<cytnx::cytnx_complex128>({1, 15, 7, 1, 15}),
            cytnx::vec_concatenate(c128v, c128v2));
  EXPECT_EQ(std::vector<cytnx::cytnx_complex64>({1, 15, 7, 1, 15}),
            cytnx::vec_concatenate(c64v, c64v2));
  EXPECT_EQ(std::vector<cytnx::cytnx_bool>({1, 0, 1, 0, 0}), cytnx::vec_concatenate(boolv, boolv2));

  cytnx::vec_concatenate_(ui64v3, ui64v, ui64v2);
  cytnx::vec_concatenate_(ui32v3, ui32v, ui32v2);
  cytnx::vec_concatenate_(ui16v3, ui16v, ui16v2);
  cytnx::vec_concatenate_(i64v3, i64v, i64v2);
  cytnx::vec_concatenate_(i32v3, i32v, i32v2);
  cytnx::vec_concatenate_(i16v3, i16v, i16v2);
  cytnx::vec_concatenate_(doublev3, doublev, doublev2);
  cytnx::vec_concatenate_(floatv3, floatv, floatv2);
  cytnx::vec_concatenate_(c128v3, c128v, c128v2);
  cytnx::vec_concatenate_(c64v3, c64v, c64v2);
  cytnx::vec_concatenate_(boolv3, boolv, boolv2);
  EXPECT_EQ(std::vector<cytnx::cytnx_uint64>({1, 15, 7, 1, 15}), ui64v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_uint32>({1, 15, 7, 1, 15}), ui32v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_uint16>({1, 15, 7, 1, 15}), ui16v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_int64>({1, 15, 7, 1, 15}), i64v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_int32>({1, 15, 7, 1, 15}), i32v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_int16>({1, 15, 7, 1, 15}), i16v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_double>({1, 15, 7, 1, 15}), doublev3);
  EXPECT_EQ(std::vector<cytnx::cytnx_float>({1, 15, 7, 1, 15}), floatv3);
  EXPECT_EQ(std::vector<cytnx::cytnx_complex128>({1, 15, 7, 1, 15}), c128v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_complex64>({1, 15, 7, 1, 15}), c64v3);
  EXPECT_EQ(std::vector<cytnx::cytnx_bool>({1, 0, 1, 0, 0}), boolv3);
}
