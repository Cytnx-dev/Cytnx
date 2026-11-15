#include <fstream>
#include <limits>
#include <string>

#include <gtest/gtest.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "Device.hpp"
#include "random.hpp"
#include "Tensor.hpp"

namespace cytnx {
  namespace test {
    namespace {

      // Threads in this process, from /proc/self/status, or -1 where that is unavailable.
      int process_thread_count() {
        std::ifstream status("/proc/self/status");
        std::string key;
        while (status >> key) {
          if (key == "Threads:") {
            int count = -1;
            status >> count;
            return count;
          }
          status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        return -1;
      }

      // A CPU transpose must not use more threads than OpenMP allows (#939); with one OpenMP
      // thread it must not start any. HPTT used to be given std::thread::hardware_concurrency()
      // threads, an explicit count that overrides OMP_NUM_THREADS, omp_set_num_threads() and the
      // process's CPU affinity. ctest runs each test in its own process, so no OpenMP thread pool
      // exists before the transpose; when the whole binary runs in one process, an earlier test may
      // already have started one and this check cannot see the difference.
      TEST(TensorTransposeThreads, RespectsOpenmpThreadCount) {
#ifndef _OPENMP
        GTEST_SKIP() << "Built without OpenMP, so without HPTT.";
#else
        if (process_thread_count() < 0) GTEST_SKIP() << "/proc/self/status is not available.";
        const int saved_threads = omp_get_max_threads();
        omp_set_num_threads(1);
        // 16^3 elements: above the 64-element cutoff below which Cytnx skips HPTT.
        const Tensor source = random::uniform({16, 16, 16}, -1.0, 1.0, Device.cpu, /*seed=*/0);
        const int threads_before = process_thread_count();
        const Tensor result = source.permute({2, 1, 0}).contiguous();
        const int threads_after = process_thread_count();
        omp_set_num_threads(saved_threads);

        EXPECT_EQ(threads_after, threads_before);
        EXPECT_EQ(result.shape(), source.shape());
        EXPECT_DOUBLE_EQ(result.at<double>({1, 2, 3}), source.at<double>({3, 2, 1}));
        EXPECT_DOUBLE_EQ(result.at<double>({15, 0, 7}), source.at<double>({7, 0, 15}));
#endif
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
