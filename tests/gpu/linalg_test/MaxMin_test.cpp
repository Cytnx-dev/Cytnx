#include <algorithm>
#include <complex>
#include <cstddef>
#include <string>
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
      Tensor make_gpu_vector(const std::vector<T>& values) {
        Tensor out({values.size()}, Type.cy_typeid_v<T>);
        std::copy(values.begin(), values.end(), out.ptr_as<T>());
        return out.to(Device.cuda);
      }

      template <typename T>
      void expect_scalar(const Tensor& result, T expected) {
        EXPECT_EQ(result.dtype(), Type.cy_typeid_v<T>);
        EXPECT_EQ(result.device(), Device.cuda);
        EXPECT_EQ(result.rank(), 0u);
        EXPECT_EQ(result.item<T>(), expected);
      }

      // The reduction kernel works on blocks of 512 elements and feeds the per-block results to
      // further passes, so these sizes cover a partly filled block, exactly one block, a second
      // and a third pass (512 * 512 = 262144 elements fill the second pass exactly).
      const std::vector<std::size_t> kSizes = {1,    2,    3,    31,     32,     33,    63,
                                               64,   65,   511,  512,    513,    1023,  1024,
                                               1025, 2000, 4096, 262143, 262144, 262145};

      // Positions, as a fraction of the size, where the single largest or smallest element is put.
      std::vector<std::size_t> extreme_positions(std::size_t size) {
        std::vector<std::size_t> positions = {0, size / 3, size / 2, (2 * size) / 3, size - 1};
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
      void check_single_extreme(T filler, T larger, T smaller) {
        for (std::size_t size : kSizes) {
          for (std::size_t position : extreme_positions(size)) {
            SCOPED_TRACE("size " + std::to_string(size) + ", position " + std::to_string(position));
            std::vector<T> values(size, filler);
            values[position] = larger;
            const Tensor with_larger = make_gpu_vector(values);
            expect_scalar<T>(linalg::Max(with_larger), larger);
            expect_scalar<T>(linalg::Min(with_larger), size > 1 ? filler : larger);

            values[position] = smaller;
            const Tensor with_smaller = make_gpu_vector(values);
            expect_scalar<T>(linalg::Min(with_smaller), smaller);
            expect_scalar<T>(linalg::Max(with_smaller), size > 1 ? filler : smaller);
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
        check_single_extreme<T>(T(-5), T(-2), T(-9));
      }

      // All values positive: a zero-filled reduction slot must not win the minimum.
      TYPED_TEST(LinalgMaxMinGpuSignedTest, SingleExtremeAllPositive) {
        using T = TypeParam;
        check_single_extreme<T>(T(5), T(9), T(2));
      }

      // Ascending values put the maximum in the last slot of the last block, the element the
      // previous in-block reduction lost.
      TYPED_TEST(LinalgMaxMinGpuSignedTest, AscendingAndDescending) {
        using T = TypeParam;
        for (std::size_t size : {1, 2, 63, 64, 65, 511, 512, 513, 2000, 4096}) {
          SCOPED_TRACE("size " + std::to_string(size));
          std::vector<T> ascending(size);
          for (std::size_t i = 0; i < size; ++i) ascending[i] = static_cast<T>(i + 1);
          const std::vector<T> descending(ascending.rbegin(), ascending.rend());
          for (const std::vector<T>& values : {ascending, descending}) {
            const Tensor tensor = make_gpu_vector(values);
            expect_scalar<T>(linalg::Max(tensor), static_cast<T>(size));
            expect_scalar<T>(linalg::Min(tensor), T(1));
          }
        }
      }

      template <typename T>
      class LinalgMaxMinGpuUnsignedTest : public ::testing::Test {};
      using UnsignedTypes = ::testing::Types<cytnx_uint64, cytnx_uint32, cytnx_uint16>;
      TYPED_TEST_SUITE(LinalgMaxMinGpuUnsignedTest, UnsignedTypes);

      TYPED_TEST(LinalgMaxMinGpuUnsignedTest, SingleExtreme) {
        using T = TypeParam;
        check_single_extreme<T>(T(5), T(9), T(2));
      }

      template <typename T>
      class LinalgMaxMinGpuComplexTest : public ::testing::Test {};
      using ComplexTypes = ::testing::Types<cytnx_complex128, cytnx_complex64>;
      TYPED_TEST_SUITE(LinalgMaxMinGpuComplexTest, ComplexTypes);

      // Complex elements are ordered by their real part; the imaginary parts here are chosen so
      // that ordering by imaginary part or by magnitude would give a different element.
      TYPED_TEST(LinalgMaxMinGpuComplexTest, SingleExtremeAllNegativeRealPart) {
        using T = TypeParam;
        check_single_extreme<T>(T(-5, 1), T(-2, -30), T(-9, 0.5));
      }

      TYPED_TEST(LinalgMaxMinGpuComplexTest, SingleExtremeAllPositiveRealPart) {
        using T = TypeParam;
        check_single_extreme<T>(T(5, 1), T(9, -30), T(2, 40));
      }

      // Among elements with equal real part the last one is returned, as on the CPU.
      TYPED_TEST(LinalgMaxMinGpuComplexTest, EqualRealPartReturnsLast) {
        using T = TypeParam;
        for (std::size_t size : {4, 64, 513, 2000}) {
          SCOPED_TRACE("size " + std::to_string(size));
          std::vector<T> values(size, T(1, 7));
          values[1] = T(3, 1);
          values[size / 2] = T(3, -4);
          const Tensor max_input = make_gpu_vector(values);
          expect_scalar<T>(linalg::Max(max_input), T(3, -4));

          values[1] = T(-3, 1);
          values[size / 2] = T(-3, -4);
          const Tensor min_input = make_gpu_vector(values);
          expect_scalar<T>(linalg::Min(min_input), T(-3, -4));
        }
      }

      // 512^3 + 1 elements need three intermediate reduction levels, so the third one reuses the
      // buffer of the first. Int16 keeps the input at about 256 MB.
      TEST(LinalgMaxMinGpuLargeTest, ThreeIntermediateLevels) {
        const std::size_t size = std::size_t{512} * 512 * 512 + 1;
        for (std::size_t position : {std::size_t{0}, size / 2, size - 1}) {
          SCOPED_TRACE("position " + std::to_string(position));
          std::vector<cytnx_int16> values(size, cytnx_int16{-5});
          values[position] = -2;
          const Tensor with_larger = make_gpu_vector(values);
          expect_scalar<cytnx_int16>(linalg::Max(with_larger), -2);
          expect_scalar<cytnx_int16>(linalg::Min(with_larger), -5);

          values[position] = -9;
          const Tensor with_smaller = make_gpu_vector(values);
          expect_scalar<cytnx_int16>(linalg::Min(with_smaller), -9);
          expect_scalar<cytnx_int16>(linalg::Max(with_smaller), -5);
        }
      }

      TEST(LinalgMaxMinGpuBoolTest, SingleExtreme) {
        for (std::size_t size : {1, 2, 64, 513, 2000}) {
          SCOPED_TRACE("size " + std::to_string(size));
          std::vector<cytnx_bool> all_false(size, false);
          expect_scalar<cytnx_bool>(linalg::Max(make_gpu_vector(all_false)), false);
          all_false[size - 1] = true;
          expect_scalar<cytnx_bool>(linalg::Max(make_gpu_vector(all_false)), true);

          std::vector<cytnx_bool> all_true(size, true);
          expect_scalar<cytnx_bool>(linalg::Min(make_gpu_vector(all_true)), true);
          all_true[size - 1] = false;
          expect_scalar<cytnx_bool>(linalg::Min(make_gpu_vector(all_true)), false);
        }
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
