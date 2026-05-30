#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_PAIRWISE_SUM_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_PAIRWISE_SUM_H_

#include <cstddef>
#include <iterator>
#include <ranges>

namespace cytnx {
  namespace linalg_internal {

    // Recursive (divide-and-conquer) core of the pairwise summation, matching
    // NumPy's np.add.reduce: a straight loop for the smallest blocks, an
    // eight-accumulator unrolled loop up to 128 elements, and a split into two
    // halves (rounded to a multiple of eight) above that. Worst-case rounding
    // error grows as O(log N * eps) instead of the O(N * eps) of a naive serial
    // accumulation, at essentially the same cost.
    template <class T, std::random_access_iterator It>
    T PairwiseSumBlocks(It first, std::size_t n) {
      if (n < 8) {
        T res = T(0);
        for (std::size_t i = 0; i < n; ++i) res += first[static_cast<std::ptrdiff_t>(i)];
        return res;
      }
      if (n <= 128) {
        T r0 = first[0], r1 = first[1], r2 = first[2], r3 = first[3];
        T r4 = first[4], r5 = first[5], r6 = first[6], r7 = first[7];
        std::size_t i = 8;
        for (; i + 8 <= n; i += 8) {
          auto p = first + static_cast<std::ptrdiff_t>(i);
          r0 += p[0];
          r1 += p[1];
          r2 += p[2];
          r3 += p[3];
          r4 += p[4];
          r5 += p[5];
          r6 += p[6];
          r7 += p[7];
        }
        T res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
        for (; i < n; ++i) res += first[static_cast<std::ptrdiff_t>(i)];
        return res;
      }
      std::size_t half = n / 2;
      half -= half % 8;
      return PairwiseSumBlocks<T>(first, half) +
             PairwiseSumBlocks<T>(first + static_cast<std::ptrdiff_t>(half), n - half);
    }

    // Pairwise sum over a random-access range. The element type is deduced from
    // the range. A contiguous std::span sums every element; pass a strided view
    // (see stride_view.hpp) to sum a strided sequence such as a matrix diagonal.
    template <std::ranges::random_access_range R>
    std::ranges::range_value_t<R> PairwiseSum(R&& range) {
      using T = std::ranges::range_value_t<R>;
      return PairwiseSumBlocks<T>(std::ranges::begin(range),
                                  static_cast<std::size_t>(std::ranges::size(range)));
    }

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_PAIRWISE_SUM_H_
