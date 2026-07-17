#include "vec_test.h"

namespace cytnx {
  namespace {
    using test::VecTest;

    /*
     * Complex types and bool types are not avaiable for vec_unique
     * Because complex types can't be compaired.
     */

    TEST_F(VecTest, VecUnique) {
      EXPECT_EQ(std::vector<cytnx_uint64>(), vec_unique(ui64v));
      EXPECT_EQ(std::vector<cytnx_uint32>(), vec_unique(ui32v));
      EXPECT_EQ(std::vector<cytnx_uint16>(), vec_unique(ui16v));
      EXPECT_EQ(std::vector<cytnx_int64>(), vec_unique(i64v));
      EXPECT_EQ(std::vector<cytnx_int32>(), vec_unique(i32v));
      EXPECT_EQ(std::vector<cytnx_int16>(), vec_unique(i16v));
      EXPECT_EQ(std::vector<cytnx_double>(), vec_unique(doublev));
      EXPECT_EQ(std::vector<cytnx_float>(), vec_unique(floatv));

      ui64v.insert(ui64v.end(), {1, 15, 15, 7, 1, 990, 423});
      ui32v.insert(ui32v.end(), {1, 15, 15, 7, 1, 990, 423});
      ui16v.insert(ui16v.end(), {1, 15, 15, 7, 1, 990, 423});
      i64v.insert(i64v.end(), {1, 15, 15, 7, 1, 990, 423});
      i32v.insert(i32v.end(), {1, 15, 15, 7, 1, 990, 423});
      i16v.insert(i16v.end(), {1, 15, 15, 7, 1, 990, 423});
      doublev.insert(doublev.end(), {1, 15, 15, 7, 1, 990, 423});
      floatv.insert(floatv.end(), {1, 15, 15, 7, 1, 990, 423});

      EXPECT_EQ(std::vector<cytnx_uint64>({1, 7, 15, 423, 990}), vec_unique(ui64v));
      EXPECT_EQ(std::vector<cytnx_uint32>({1, 7, 15, 423, 990}), vec_unique(ui32v));
      EXPECT_EQ(std::vector<cytnx_uint16>({1, 7, 15, 423, 990}), vec_unique(ui16v));
      EXPECT_EQ(std::vector<cytnx_int64>({1, 7, 15, 423, 990}), vec_unique(i64v));
      EXPECT_EQ(std::vector<cytnx_int32>({1, 7, 15, 423, 990}), vec_unique(i32v));
      EXPECT_EQ(std::vector<cytnx_int16>({1, 7, 15, 423, 990}), vec_unique(i16v));
      EXPECT_EQ(std::vector<cytnx_double>({1, 7, 15, 423, 990}), vec_unique(doublev));
      EXPECT_EQ(std::vector<cytnx_float>({1, 7, 15, 423, 990}), vec_unique(floatv));
    }

  }  // namespace
}  // namespace cytnx
