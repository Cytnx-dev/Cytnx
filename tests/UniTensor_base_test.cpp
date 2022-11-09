#include "UniTensor_base_test.h"

TEST_F(UniTensor_baseTest, get_index) {
  utzero345.set_labels({"abc", "ABC", "CBA"});
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
