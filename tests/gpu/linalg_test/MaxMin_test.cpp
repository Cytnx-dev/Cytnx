#include <algorithm>
#include <complex>
#include <vector>

#include "gtest/gtest.h"

#include "Device.hpp"
#include "linalg.hpp"
#include "Tensor.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace gpu_test {
    namespace {

      template <typename T>
      Tensor MakeGpuVector(const std::vector<T>& values) {
        Tensor out({values.size()}, Type.cy_typeid_v<T>);
        std::copy(values.begin(), values.end(), out.ptr_as<T>());
        return out.to(Device.cuda);
      }

      template <typename T>
      void ExpectScalar(const Tensor& result, T expected) {
        EXPECT_EQ(result.dtype(), Type.cy_typeid_v<T>);
        EXPECT_EQ(result.device(), Device.cuda);
        EXPECT_EQ(result.rank(), 0u);
        EXPECT_EQ(result.item<T>(), expected);
      }

      // The reduction kernel works on blocks of 512 elements and feeds the per-block results to
      // further passes, so these sizes cover a partly filled block, exactly one block, a second
      // and a third pass (512 * 512 = 262144 elements fill the second pass exactly).
      const std::vector<cytnx_uint64> kSizes = {1,    2,    3,    31,     32,     33,    63,
                                                64,   65,   511,  512,    513,    1023,  1024,
                                                1025, 2000, 4096, 262143, 262144, 262145};

      // Positions, as a fraction of the size, where the single largest or smallest element is put.
      std::vector<cytnx_uint64> ExtremePositions(cytnx_uint64 size) {
        std::vector<cytnx_uint64> positions = {0, size / 3, size / 2, (2 * size) / 3, size - 1};
        if (size > 42) positions.push_back(42);
        if (size > 513) positions.push_back(513);
        std::sort(positions.begin(), positions.end());
        positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
        return positions;
      }

      // Every element equals `filler` except one, `extreme`, at each tested position. The expected
      // result is `extreme` by construction when it is the wanted extremum and `filler` when it is
      // the opposite one (for sizes above 1).
      template <typename T>
      void CheckSingleExtreme(T filler, T larger, T smaller) {
        for (cytnx_uint64 size : kSizes) {
          for (cytnx_uint64 position : ExtremePositions(size)) {
            SCOPED_TRACE("size " + std::to_string(size) + ", position " + std::to_string(position));
            std::vector<T> values(size, filler);
            values[position] = larger;
            const Tensor with_larger = MakeGpuVector(values);
            ExpectScalar<T>(linalg::Max(with_larger), larger);
            ExpectScalar<T>(linalg::Min(with_larger), size > 1 ? filler : larger);

            values[position] = smaller;
            const Tensor with_smaller = MakeGpuVector(values);
            ExpectScalar<T>(linalg::Min(with_smaller), smaller);
            ExpectScalar<T>(linalg::Max(with_smaller), size > 1 ? filler : smaller);
          }
        }
      }

      template <typename T>
      class LinalgMaxMinGpuSignedTest : public ::testing::Test {};
      using SignedRealTypes =
        ::testing::Types<cytnx_double, cytnx_float, cytnx_int64, cytnx_int32, cytnx_int16>;
      TYPED_TEST_SUITE(LinalgMaxMinGpuSignedTest, SignedRealTypes);

      // All values negative: a zero-filled reduction slot must not win the maximum.
      TYPED_TEST(LinalgMaxMinGpuSignedTest, SingleExtremeAllNegative) {
        using T = TypeParam;
        CheckSingleExtreme<T>(T(-5), T(-2), T(-9));
      }

      // All values positive: a zero-filled reduction slot must not win the minimum.
      TYPED_TEST(LinalgMaxMinGpuSignedTest, SingleExtremeAllPositive) {
        using T = TypeParam;
        CheckSingleExtreme<T>(T(5), T(9), T(2));
      }

      // Ascending values put the maximum in the last slot of the last block, the element the
      // previous in-block reduction lost.
      TYPED_TEST(LinalgMaxMinGpuSignedTest, AscendingAndDescending) {
        using T = TypeParam;
        for (cytnx_uint64 size : {1, 2, 63, 64, 65, 511, 512, 513, 2000, 4096}) {
          SCOPED_TRACE("size " + std::to_string(size));
          std::vector<T> ascending(size);
          for (cytnx_uint64 i = 0; i < size; ++i) ascending[i] = static_cast<T>(i + 1);
          const std::vector<T> descending(ascending.rbegin(), ascending.rend());
          for (const std::vector<T>& values : {ascending, descending}) {
            const Tensor tensor = MakeGpuVector(values);
            ExpectScalar<T>(linalg::Max(tensor), static_cast<T>(size));
            ExpectScalar<T>(linalg::Min(tensor), T(1));
          }
        }
      }

      template <typename T>
      class LinalgMaxMinGpuUnsignedTest : public ::testing::Test {};
      using UnsignedTypes = ::testing::Types<cytnx_uint64, cytnx_uint32, cytnx_uint16>;
      TYPED_TEST_SUITE(LinalgMaxMinGpuUnsignedTest, UnsignedTypes);

      TYPED_TEST(LinalgMaxMinGpuUnsignedTest, SingleExtreme) {
        using T = TypeParam;
        CheckSingleExtreme<T>(T(5), T(9), T(2));
      }

      template <typename T>
      class LinalgMaxMinGpuComplexTest : public ::testing::Test {};
      using ComplexTypes = ::testing::Types<cytnx_complex128, cytnx_complex64>;
      TYPED_TEST_SUITE(LinalgMaxMinGpuComplexTest, ComplexTypes);

      // Complex elements are ordered by their real part; the imaginary parts here are chosen so
      // that ordering by imaginary part or by magnitude would give a different element.
      TYPED_TEST(LinalgMaxMinGpuComplexTest, SingleExtremeAllNegativeRealPart) {
        using T = TypeParam;
        CheckSingleExtreme<T>(T(-5, 1), T(-2, -30), T(-9, 0.5));
      }

      TYPED_TEST(LinalgMaxMinGpuComplexTest, SingleExtremeAllPositiveRealPart) {
        using T = TypeParam;
        CheckSingleExtreme<T>(T(5, 1), T(9, -30), T(2, 40));
      }

      // Among elements with equal real part the last one is returned, as on the CPU.
      TYPED_TEST(LinalgMaxMinGpuComplexTest, EqualRealPartReturnsLast) {
        using T = TypeParam;
        for (cytnx_uint64 size : {4, 64, 513, 2000}) {
          SCOPED_TRACE("size " + std::to_string(size));
          std::vector<T> values(size, T(1, 7));
          values[1] = T(3, 1);
          values[size / 2] = T(3, -4);
          const Tensor max_input = MakeGpuVector(values);
          ExpectScalar<T>(linalg::Max(max_input), T(3, -4));

          values[1] = T(-3, 1);
          values[size / 2] = T(-3, -4);
          const Tensor min_input = MakeGpuVector(values);
          ExpectScalar<T>(linalg::Min(min_input), T(-3, -4));
        }
      }

      TEST(LinalgMaxMinGpuBoolTest, SingleExtreme) {
        for (cytnx_uint64 size : {1, 2, 64, 513, 2000}) {
          SCOPED_TRACE("size " + std::to_string(size));
          std::vector<cytnx_bool> all_false(size, false);
          ExpectScalar<cytnx_bool>(linalg::Max(MakeGpuVector(all_false)), false);
          all_false[size - 1] = true;
          ExpectScalar<cytnx_bool>(linalg::Max(MakeGpuVector(all_false)), true);

          std::vector<cytnx_bool> all_true(size, true);
          ExpectScalar<cytnx_bool>(linalg::Min(MakeGpuVector(all_true)), true);
          all_true[size - 1] = false;
          ExpectScalar<cytnx_bool>(linalg::Min(MakeGpuVector(all_true)), false);
        }
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
