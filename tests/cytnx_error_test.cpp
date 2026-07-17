#include <cstdarg>
#include <limits>
#include <stdexcept>
#include <string>

#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "utils/checked_cast.hpp"
namespace cytnx {
  namespace {
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
      EXPECT_THROW(cytnx_error_msg(true, "boom%s", ""), error);
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
      } catch (const error &e) {
        EXPECT_EQ(std::string(e.what()).find("%s"), std::string::npos);
      }
    }

    TEST(CytnxError, ZeroVarargCallCompilesAndThrows) {
      EXPECT_THROW(cytnx_error_msg(true, "plain message"), error);
    }

    TEST(CytnxError, WarningDoesNotThrow) {
      EXPECT_NO_THROW(cytnx_warning_msg(true, "just a warning%s", ""));
    }

    TEST(CytnxError, PercentLiteralFormatsCorrectly) {
      try {
        cytnx_error_msg(true, "100%%");
        FAIL() << "expected throw";
      } catch (const error &e) {
        EXPECT_NE(std::string(e.what()).find("100%"), std::string::npos);
        EXPECT_EQ(std::string(e.what()).find("100%%"), std::string::npos);
      }
    }

    namespace {
      std::string CallVformat(const char *fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        std::string s = internal::vformat_message(fmt, ap);
        va_end(ap);
        return s;
      }
    }  // namespace

    // A null format must not be dereferenced (vsnprintf(nullptr,...)/std::string(nullptr) are UB).
    TEST(CytnxError, NullFormatDoesNotCrash) { EXPECT_EQ(CallVformat(nullptr), std::string()); }

    // Report composition must tolerate null kind/func/file (they are non-null in practice, but a
    // report composer should never itself be UB).
    TEST(CytnxError, ComposeReportToleratesNulls) {
      std::string r = internal::compose_report(nullptr, nullptr, nullptr, 42, "boom");
      EXPECT_NE(r.find("boom"), std::string::npos);
      EXPECT_NE(r.find("error"), std::string::npos);  // kind fallback
      EXPECT_NE(r.find("unknown"), std::string::npos);  // func / file fallback
    }

    // A null `name` must not reach the %s conversion; it throws (not crashes) on overflow.
    TEST(CytnxError, CheckedCastNullNameDoesNotCrash) {
      EXPECT_NO_THROW(internal::CheckedCastToInt64(5, nullptr));
      const cytnx_uint64 too_big =
        static_cast<cytnx_uint64>(std::numeric_limits<cytnx_int64>::max()) + 1;
      EXPECT_THROW(internal::CheckedCastToInt64(too_big, nullptr), std::logic_error);
    }

    // Single-report: an error throws only -- with User_debug off it writes nothing to stderr
    // (the backtrace is gated behind User_debug), so throw-heavy suites don't spam the logs.
    TEST(CytnxError, ErrorDoesNotEchoToStderr) {
      const bool prev = User_debug;
      User_debug = false;
      testing::internal::CaptureStderr();
      EXPECT_THROW(cytnx_error_msg(true, "boom%s", ""), error);
      const std::string captured = testing::internal::GetCapturedStderr();
      User_debug = prev;
      EXPECT_TRUE(captured.empty()) << "error unexpectedly wrote to stderr: " << captured;
    }

    // Warnings do not throw and are reported to stderr.
    TEST(CytnxError, WarningEchoesToStderr) {
      testing::internal::CaptureStderr();
      cytnx_warning_msg(true, "heads up%s", "");
      const std::string captured = testing::internal::GetCapturedStderr();
      EXPECT_NE(captured.find("heads up"), std::string::npos);
      EXPECT_NE(captured.find("Cytnx warning"), std::string::npos);
    }

    // The macro-expanded report carries the call-site function, file, and line.
    TEST(CytnxError, ReportCarriesFuncFileLine) {
      try {
        cytnx_error_msg(true, "located%s", "");
        FAIL() << "expected throw";
      } catch (const error &e) {
        const std::string w = e.what();
        EXPECT_NE(w.find("located"), std::string::npos);  // the message
        EXPECT_NE(w.find(" occur at "), std::string::npos);  // function section
        EXPECT_NE(w.find("ReportCarriesFuncFileLine"), std::string::npos);  // CYTNX_FUNC_NAME
        EXPECT_NE(w.find("# file : "), std::string::npos);  // file section
        EXPECT_NE(w.find("cytnx_error_test"), std::string::npos);  // __FILE__
      }
    }

  }  // namespace
}  // namespace cytnx
