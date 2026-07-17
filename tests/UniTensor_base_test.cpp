#include "UniTensor_base_test.h"

namespace cytnx {
  namespace test {

    TEST_F(UniTensor_baseTest, GetIndex) {
      utzero345.relabel_({"abc", "ABC", "CBA"});
      EXPECT_EQ(utzero345.get_index("abc"), 0);
      EXPECT_EQ(utzero345.get_index("ABC"), 1);
      EXPECT_EQ(utzero345.get_index("CBA"), 2);
      EXPECT_EQ(utzero345.get_index("ABCa"), -1);
      EXPECT_EQ(utzero345.get_index(""), -1);
      std::cout << utzero345.get_index("abc") << std::endl;
      std::cout << utzero345.get_index("ABC") << std::endl;
      std::cout << utzero345.get_index("CBA") << std::endl;
      std::cout << utzero345.get_index("ABCa") << std::endl;
      std::cout << utzero345.get_index("") << std::endl;
    }

    TEST_F(UniTensor_baseTest, RankDetectsLabelBondMismatch) {
      ASSERT_EQ(utzero345.rank(), 3);
      // rank() guards against an internal labels/bonds size mismatch. Since #1001 made bonds()
      // immutable, that desync is unreachable through the public API by design, so poke the
      // internal vector directly (const_cast on the reference to _bonds) to exercise the guard.
      const_cast<std::vector<Bond> &>(utzero345.bonds()).pop_back();
      EXPECT_THROW(utzero345.rank(), error);
    }

  }  // namespace test
}  // namespace cytnx
