// Regression guard for #951: cytnx public headers must stay usable in a TU
// that sees the C complex.h macro `I` before including any cytnx header.
#include <complex.h>
#ifndef I
  // libc++'s complex.h does not leak the C macro `I` in C++ mode; define it
  // explicitly so this guard bites on every platform, not just glibc.
  #define I (18973)
#endif

#include "cytnx.hpp"

#undef I

#include "gtest/gtest.h"

namespace cytnx {
  namespace test {

    TEST(IncludeOrderCanary, HeadersSurviveComplexHMacroI) { SUCCEED(); }

  }  // namespace test
}  // namespace cytnx
