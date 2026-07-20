#include <vector>

#include "gtest/gtest.h"

#include "cytnx.hpp"

// Regression guard for #1003 (Ian's note): the namespace-scope complex<->builtin operator
// overloads in utils/complex_arithmetic.hpp leaked into ordinary overload resolution for unrelated
// code. Because cytnx_complex64/128 are aliases for std::complex<float/double>, and builtins
// convert into them, expressions on plain std/builtin types picked up the cytnx operators as viable
// candidates -- amplified by C++20's reversed operator== rewrite. The classic symptom was that
// `std::vector<bool>::reference == bool` became ambiguous under `using namespace cytnx`.
//
// This is primarily a COMPILE-TIME guard: if those namespace-scope operators are reintroduced, this
// translation unit stops compiling. The runtime assertions just pin the expected boolean results.
using namespace cytnx;

TEST(OverloadHygiene, VectorBoolReferenceComparesWithBool) {
  std::vector<bool> values(2);
  values[0] = true;
  values[1] = false;
  bool rhs = true;

  // The offending expression from the original report.
  EXPECT_TRUE(values[0] == rhs);
  EXPECT_FALSE(values[1] == rhs);

  // Reversed operand order (also went through the reversed-candidate path).
  EXPECT_TRUE(rhs == values[0]);
  EXPECT_FALSE(rhs == values[1]);
}

TEST(OverloadHygiene, BuiltinEqualityIsUnambiguous) {
  // Plain builtin comparisons must not be hijacked by cytnx complex overloads either.
  int i = 3;
  double d = 3.0;
  bool b = true;
  EXPECT_TRUE(i == 3);
  EXPECT_TRUE(d == 3.0);
  EXPECT_TRUE(b == true);
}
