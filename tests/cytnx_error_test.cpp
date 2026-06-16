#include <gtest/gtest.h>

#include "cytnx_error.hpp"

#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

  class ScopedStackTraceEnv {
   public:
    explicit ScopedStackTraceEnv(const char *value) {
      const char *old_value = std::getenv("CYTNX_STACKTRACE");
      if (old_value != nullptr) {
        old_value_ = old_value;
      }
      setenv("CYTNX_STACKTRACE", value, 1);
    }

    ~ScopedStackTraceEnv() {
      if (old_value_.has_value()) {
        setenv("CYTNX_STACKTRACE", old_value_->c_str(), 1);
      } else {
        unsetenv("CYTNX_STACKTRACE");
      }
    }

    ScopedStackTraceEnv(const ScopedStackTraceEnv &) = delete;
    ScopedStackTraceEnv &operator=(const ScopedStackTraceEnv &) = delete;

   private:
    std::optional<std::string> old_value_;
  };

  TEST(CytnxErrorTest, ErrorMessageSupportsLongFormattedText) {
    const std::string payload(2000, 'x');

    testing::internal::CaptureStderr();
    try {
      cytnx_error("long message: %s", payload.c_str());
      FAIL() << "Expected cytnx_error to throw.";
    } catch (const std::logic_error &err) {
      const std::string stderr_output = testing::internal::GetCapturedStderr();
      const std::string message = err.what();
      EXPECT_NE(message.find("# Cytnx error occur at"), std::string::npos);
      EXPECT_NE(message.find("long message: " + payload), std::string::npos);
      EXPECT_NE(stderr_output.find("long message: " + payload), std::string::npos);
    }
  }

  TEST(CytnxErrorTest, ErrorMessageDoesNotNeedDummyVarargs) {
    testing::internal::CaptureStderr();
    try {
      cytnx_error_if(true, "plain message without dummy varargs");
      FAIL() << "Expected cytnx_error_if to throw.";
    } catch (const std::logic_error &err) {
      testing::internal::GetCapturedStderr();
      const std::string message = err.what();
      EXPECT_NE(message.find("plain message without dummy varargs"), std::string::npos);
    }
  }

  TEST(CytnxErrorTest, WarningMessageSupportsLongFormattedText) {
    const std::string payload(2000, 'w');

    testing::internal::CaptureStderr();
    cytnx_warning("long warning: %s", payload.c_str());
    const std::string stderr_output = testing::internal::GetCapturedStderr();

    EXPECT_NE(stderr_output.find("# Cytnx warning occur at"), std::string::npos);
    EXPECT_NE(stderr_output.find("long warning: " + payload), std::string::npos);
  }

  TEST(CytnxErrorTest, WarningMessageDoesNotNeedDummyVarargs) {
    testing::internal::CaptureStderr();
    cytnx_warning_if(true, "plain warning without dummy varargs");
    const std::string stderr_output = testing::internal::GetCapturedStderr();

    EXPECT_NE(stderr_output.find("plain warning without dummy varargs"), std::string::npos);
  }

  TEST(CytnxErrorTest, ConditionalAliasesDoNothingWhenFalse) {
    testing::internal::CaptureStderr();
    cytnx_error_if(false, "suppressed error");
    cytnx_warning_if(false, "suppressed warning");
    cytnx_error_msg(false, "suppressed legacy error");
    cytnx_warning_msg(false, "suppressed legacy warning");
    const std::string stderr_output = testing::internal::GetCapturedStderr();

    EXPECT_TRUE(stderr_output.empty());
  }

  TEST(CytnxErrorTest, StackTraceEnvironmentParserIsCaseInsensitive) {
    EXPECT_FALSE(cytnx_error_detail::stack_trace_setting_enabled("Off"));
    EXPECT_FALSE(cytnx_error_detail::stack_trace_setting_enabled("FALSE"));
    EXPECT_FALSE(cytnx_error_detail::stack_trace_setting_enabled("No"));
    EXPECT_TRUE(cytnx_error_detail::stack_trace_setting_enabled("1"));
    EXPECT_TRUE(cytnx_error_detail::stack_trace_setting_enabled("on"));
  }

  TEST(CytnxErrorTest, StackTraceCanBeDisabledByEnvironment) {
    ScopedStackTraceEnv stack_trace("Off");

    testing::internal::CaptureStderr();
    try {
      cytnx_error("stack trace disabled");
      FAIL() << "Expected cytnx_error to throw.";
    } catch (const std::logic_error &err) {
      const std::string stderr_output = testing::internal::GetCapturedStderr();
      const std::string message = err.what();
      EXPECT_NE(message.find("stack trace disabled"), std::string::npos);
      EXPECT_NE(stderr_output.find("stack trace disabled"), std::string::npos);
      EXPECT_EQ(stderr_output.find("Stack trace"), std::string::npos);
    }
  }

}  // namespace
