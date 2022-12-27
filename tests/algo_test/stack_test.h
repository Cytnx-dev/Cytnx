#ifndef _H_stack_test
#define _H_stack_test

#include "cytnx.hpp"
#include <gtest/gtest.h>
using namespace cytnx;

namespace HstackTest {

static std::vector<unsigned int> dtype_list = {
  //Type.Void,
  Type.ComplexDouble,
  Type.ComplexFloat,
  Type.Double,
  Type.Float,
  Type.Int64,
  Type.Uint64,
  Type.Int32,
  Type.Uint32,
  Type.Int16,
  Type.Uint16,
  Type.Bool
};

static std::vector<int> device_list = {
  Device.cpu,
  //Device.cuda,  //currently cuda version still not implement
};

//init the test data uniform randomize
void InitTestData(std::vector<Tensor>& Ts);

bool IsElemSame(const Tensor& T1, const std::vector<cytnx_uint64>& idices1,
                const Tensor& T2, const std::vector<cytnx_uint64>& idices2);

} //namespace HstackTest

namespace VstackTest {
  using namespace HstackTest;
} //namespace VstackTest
#endif
