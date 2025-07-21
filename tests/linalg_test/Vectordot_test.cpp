#include "cytnx.hpp"
#include <gtest/gtest.h>
#include "../test_tools.h"

using namespace cytnx;
using namespace testing;

namespace {
  auto device = Device.cpu;
  bool CheckResult(const Tensor& l, const Tensor& r, const Tensor& out, bool is_conj) {
	if (out.shape().size() != 1 || out.shape()[0] != 1) {
	  return false;
	}
	if (l.shape().size() != 1 || r.shape().size()!= 1) {
	  return false;
	}
	auto len = l.shape()[0];
	auto tmp = l;
	tmp = is_conj ? tmp.Conj() : tmp;
	auto tmp_mul = linalg::Mul(tmp, r);
	Tensor ans = Tensor({1}, tmp_mul.dtype());
	ans.at({0}) = 0;
	for (int i = 0; i < len; ++i) {
	  ans += tmp_mul.at({i});
	}
	double tol = (tmp_mul.dtype() == Type.Float || Type.ComplexFloat) ?
			1.0e-4 : 1.0e-12;
	if (ans.at({0}) == 0) {
	  return false;
	}
	return TestTools::AreNearlyEqTensor(ans, out, tol);
  }

  Tensor InitTensor(const int len, const unsigned int dtype, const int seed = 0) {
	Tensor t;
	if (Type.is_float(dtype)) {
      double low = -1.0;
      double high = 1.0;
      t = random::uniform({len}, low, high, device, seed, dtype);
	} else {
	  t = cytnx::arange(len);
	}
	return t;
  }
}

//test is_conj = treu
TEST(Vectordot, complex128_conj) {
  double low = -1.0;
  double high = 1.0;
  unsigned int seed1 = 0;
  unsigned int seed2 = 1;
  bool is_conj = true;
  constexpr int len = 10;
  Tensor a = InitTensor(len, Type.ComplexDouble, seed1);
  Tensor b = InitTensor(len, Type.ComplexDouble, seed2);
  auto out = linalg::Vectordot(a, b, is_conj);
  bool is_correct = CheckResult(a, b, out, is_conj);
  EXPECT_TRUE(is_correct);
}

//test is_conj = false
TEST(Vectordot, complex128) {
  double low = -1.0;
  double high = 1.0;
  unsigned int seed1 = 0;
  unsigned int seed2 = 1;
  bool is_conj = false;
  constexpr int len = 10;
  Tensor a = InitTensor(len, Type.ComplexDouble, seed1);
  Tensor b = InitTensor(len, Type.ComplexDouble, seed2);
  auto out = linalg::Vectordot(a, b, is_conj);
  bool is_correct = CheckResult(a, b, out, is_conj);
  EXPECT_TRUE(is_correct);
}

//test all type, L and R type may be different.
TEST(Vectordot, alltype) {
  bool is_conj = false;
  constexpr int len = 10;
  bool is_correct = false;
  Tensor a, b;

  for (auto dtype1:TestTools::dtype_list) {
    for (auto dtype2:TestTools::dtype_list) {
	  if (dtype1 == Type.Bool && dtype2 == Type.Bool) {
	    continue;
	  } 
      a = InitTensor(len, dtype1);
      b = InitTensor(len, dtype2);
      auto out = linalg::Vectordot(a, b, is_conj);
      is_correct = CheckResult(a, b, out, is_conj);
	  if (!is_correct) 
	    break;
    }
	if (!is_correct) 
	  break;
  }
  EXPECT_TRUE(is_correct);
}

// error test
// test error dtype input, both dtype = Bool
TEST(Vectordot, err_dtype) {
  constexpr int len = 10;
  Tensor a, b;
  a = cytnx::arange(len);
  a = a.astype(Type.Bool);
  b = a;
  try {
    auto out = linalg::Vectordot(a, b);
    FAIL();
  } catch (const std::exception& ex) {
    auto err_msg = ex.what();
    std::cerr << err_msg << std::endl;
    SUCCEED();
  }
}

//test the length is not match
TEST(Vectordot, err_len_diff) {
  constexpr int len1 = 1;
  constexpr int len2 = 10;
  Tensor a, b;
  a = InitTensor(len1, Type.Double);
  a = InitTensor(len2, Type.Double);
  try {
    auto out = linalg::Vectordot(a, b);
    FAIL();
  } catch (const std::exception& ex) {
    auto err_msg = ex.what();
    std::cerr << err_msg << std::endl;
    SUCCEED();
  }
}

//test the size of shape is not 1
TEST(Vectordot, err_shape) {
  constexpr int row = 3;
  constexpr int col = 3;
  double low = -1.0;
  double high = 1.0;
  int seed = 0;
  Tensor a = random::uniform({row, col}, low, high, device, seed, Type.Double);
  Tensor b = a;
  try {
    auto out = linalg::Vectordot(a, b);
    FAIL();
  } catch (const std::exception& ex) {
    auto err_msg = ex.what();
    std::cerr << err_msg << std::endl;
    SUCCEED();
  }
}
