#include <stdexcept>
#include <string>

#include "cytnx.hpp"
#include "gtest/gtest.h"

TEST(CytnxError, LongMessagesDoNotOverflow) {
  std::string big(5000, 'x');
  try {
    cytnx_error_msg(true, "%s", big.c_str());
    FAIL() << "expected throw";
  } catch (const std::logic_error &e) {
    EXPECT_NE(std::string(e.what()).find("xxxx"), std::string::npos);
    EXPECT_GE(std::string(e.what()).size(), big.size());
  }
}

TEST(CytnxError, ThrowsCytnxErrorType) {
  EXPECT_THROW(cytnx_error_msg(true, "boom%s", ""), cytnx::error);
  EXPECT_THROW(cytnx_error_msg(true, "boom%s", ""), std::logic_error);
}

TEST(CytnxError, FalseConditionDoesNotThrow) {
  EXPECT_NO_THROW(cytnx_error_msg(false, "never%s", ""));
}

TEST(CytnxError, MacroIsSafeInIfElse) {
  bool flag = false;
  if (flag)
    cytnx_error_msg(true, "unreachable%s", "");
  else
    SUCCEED();
}

TEST(CytnxError, ConditionEvaluatedOnce) {
  int n = 0;
  cytnx_error_msg((++n, false), "never%s", "");
  EXPECT_EQ(n, 1);
}

TEST(CytnxError, EmptyFormattedMessageIsEmpty) {
  try {
    cytnx_error_msg(true, "%s", "");
    FAIL() << "expected throw";
  } catch (const cytnx::error &e) {
    EXPECT_EQ(std::string(e.what()).find("%s"), std::string::npos);
  }
}

TEST(CytnxError, ZeroVarargCallCompilesAndThrows) {
  EXPECT_THROW(cytnx_error_msg(true, "plain message"), cytnx::error);
}

TEST(CytnxError, WarningDoesNotThrow) {
  EXPECT_NO_THROW(cytnx_warning_msg(true, "just a warning%s", ""));
}

TEST(CytnxError, PercentLiteralFormatsCorrectly) {
  try {
    cytnx_error_msg(true, "100%%");
    FAIL() << "expected throw";
  } catch (const cytnx::error &e) {
    EXPECT_NE(std::string(e.what()).find("100%"), std::string::npos);
    EXPECT_EQ(std::string(e.what()).find("100%%"), std::string::npos);
  }
}
