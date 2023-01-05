#ifndef _H_stack_test
#define _H_stack_test

#include "cytnx.hpp"
#include <gtest/gtest.h>
using namespace cytnx;

namespace HstackTest {

bool IsElemSame(const Tensor& T1, const std::vector<cytnx_uint64>& idices1,
                const Tensor& T2, const std::vector<cytnx_uint64>& idices2);

}

namespace VstackTest {
  using namespace HstackTest;
}

#endif
