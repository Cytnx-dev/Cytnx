#ifndef _H_test_tools
#define _H_test_tools

#include "cytnx.hpp"
#include <gtest/gtest.h>

//this file contains some function we may usually use in the unit test 
//  such as the data initialization and comparison.

using namespace cytnx;

namespace TestTools {

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

//Tensor tools

//given the tensor T with shape and dtype has been initialzed, set its data as random uniform.
void InitTensorUniform(Tensor& T, unsigned int rand_seed = 0);
void InitTensorUniform(std::vector<Tensor>& T, unsigned int rand_seed = 0);

bool AreNearlyEqTensor(const Tensor& T1, const Tensor& T2, const cytnx_double tol = 0);
bool AreEqTensor(const Tensor& T1, const Tensor& T2);

} //namespace TestTools

#endif
