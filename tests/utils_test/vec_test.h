#ifndef CYTNX_TESTS_UTILS_TEST_VEC_TEST_H_
#define CYTNX_TESTS_UTILS_TEST_VEC_TEST_H_

#include <gtest/gtest.h>

#include "cytnx.hpp"
namespace cytnx {
  namespace test {
    class VecTest : public ::testing::Test {
     public:
      std::vector<cytnx_uint64> ui64v;
      std::vector<cytnx_uint32> ui32v;
      std::vector<cytnx_uint16> ui16v;
      std::vector<cytnx_int64> i64v;
      std::vector<cytnx_int32> i32v;
      std::vector<cytnx_int16> i16v;
      std::vector<cytnx_double> doublev;
      std::vector<cytnx_float> floatv;
      std::vector<cytnx_complex128> c128v;
      std::vector<cytnx_complex64> c64v;
      std::vector<cytnx_bool> boolv;

      std::vector<cytnx_uint64> ui64v2;
      std::vector<cytnx_uint32> ui32v2;
      std::vector<cytnx_uint16> ui16v2;
      std::vector<cytnx_int64> i64v2;
      std::vector<cytnx_int32> i32v2;
      std::vector<cytnx_int16> i16v2;
      std::vector<cytnx_double> doublev2;
      std::vector<cytnx_float> floatv2;
      std::vector<cytnx_complex128> c128v2;
      std::vector<cytnx_complex64> c64v2;
      std::vector<cytnx_bool> boolv2;

      std::vector<cytnx_uint64> ui64v3;
      std::vector<cytnx_uint32> ui32v3;
      std::vector<cytnx_uint16> ui16v3;
      std::vector<cytnx_int64> i64v3;
      std::vector<cytnx_int32> i32v3;
      std::vector<cytnx_int16> i16v3;
      std::vector<cytnx_double> doublev3;
      std::vector<cytnx_float> floatv3;
      std::vector<cytnx_complex128> c128v3;
      std::vector<cytnx_complex64> c64v3;
      std::vector<cytnx_bool> boolv3;

     protected:
      void SetUp() override {}

      void TearDown() override { ui64v.clear(); }
    };

  }  // namespace test
}  // namespace cytnx
#endif  // CYTNX_TESTS_UTILS_TEST_VEC_TEST_H_
