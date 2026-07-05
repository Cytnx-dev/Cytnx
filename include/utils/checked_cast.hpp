#ifndef CYTNX_UTILS_CHECKED_CAST_H_
#define CYTNX_UTILS_CHECKED_CAST_H_

#include <limits>
#include <source_location>

#include "Type.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace internal {

    /// @brief Narrow a @c cytnx_uint64 index to @c cytnx_int64, rejecting values that
    /// would overflow.
    /// @param value The unsigned value to convert, typically an index or extent
    /// derived from a shape/stride that is mathematically non-negative but stored
    /// in an unsigned type.
    /// @param name Identifier used in the error message if @p value overflows.
    /// @param location Defaulted to the call site, so the error reports where the
    /// narrowing was requested rather than this function's own location. Calling
    /// @c cytnx_error_msg here directly would always report this line, since the
    /// macro captures @c __FILE__ / @c __LINE__ where it is expanded, not where
    /// @c CheckedCastToInt64 is invoked.
    /// @throws std::logic_error (via @c error_msg) if @p value exceeds the range
    /// of @c cytnx_int64.
    inline cytnx_int64 CheckedCastToInt64(
      cytnx_uint64 value, const char *name,
      std::source_location location = std::source_location::current()) {
      if (value > static_cast<cytnx_uint64>(std::numeric_limits<cytnx_int64>::max())) {
        cytnx::internal::error_msg_impl(location.function_name(), location.file_name(),
                                        static_cast<int>(location.line()),
                                        "[ERROR] %s=%llu exceeds cytnx_int64 max.\n", name,
                                        static_cast<unsigned long long>(value));
      }
      return static_cast<cytnx_int64>(value);
    }

  }  // namespace internal
}  // namespace cytnx

#endif  // CYTNX_UTILS_CHECKED_CAST_H_
