#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

      // Each test reads the process's thread count before and after one transpose. ctest runs each
      // test in its own process, so no OpenMP thread pool exists beforehand; when the whole binary
      // runs in one process, an earlier test may already have started one and these checks cannot
      // see the difference.

#ifdef _OPENMP
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

      // Transposes `source` (rank 3) under the given OpenMP and Device.Ncpus limits; returns the
      // number of threads the process gained.
      int threads_started_by_transpose(const Tensor& source, int openmp_threads, int ncpus,
                                       Tensor& result) {
        const int saved_openmp_threads = omp_get_max_threads();
        const int saved_ncpus = Device.Ncpus;
        omp_set_num_threads(openmp_threads);
        Device.Ncpus = ncpus;
        const int threads_before = process_thread_count();
        result = source.permute({2, 1, 0}).contiguous();
        const int threads_after = process_thread_count();
        Device.Ncpus = saved_ncpus;
        omp_set_num_threads(saved_openmp_threads);
        return threads_after - threads_before;
      }

      void expect_reversed(const Tensor& result, const Tensor& source) {
        const auto n = source.shape()[0];
        EXPECT_EQ(result.shape(), source.shape());
        EXPECT_DOUBLE_EQ(result.at<double>({1, 2, 3}), source.at<double>({3, 2, 1}));
        EXPECT_DOUBLE_EQ(result.at<double>({n - 1, 0, 7}), source.at<double>({7, 0, n - 1}));
      }
#endif  // _OPENMP

      // A transpose must not use more threads than OpenMP allows (#939). 64^3 Double elements
      // (2 MiB) are enough work for many threads, but with one OpenMP thread it must start none.
      // HPTT used to be given std::thread::hardware_concurrency() threads, an explicit count that
      // overrides OMP_NUM_THREADS, omp_set_num_threads() and the process's CPU affinity.
      TEST(TensorTransposeThreads, RespectsOpenmpThreadCount) {
#ifndef _OPENMP
        GTEST_SKIP() << "Built without OpenMP, so without HPTT.";
#else
        if (process_thread_count() < 0) GTEST_SKIP() << "/proc/self/status is not available.";
        const Tensor source = random::uniform({64, 64, 64}, -1.0, 1.0, Device.cpu, /*seed=*/0);
        Tensor result;
        EXPECT_EQ(threads_started_by_transpose(source, /*openmp_threads=*/1, /*ncpus=*/4, result),
                  0);
        expect_reversed(result, source);
#endif
      }

      // Device.Ncpus caps the threads of a transpose too, so setting it takes effect.
      TEST(TensorTransposeThreads, RespectsDeviceNcpus) {
#ifndef _OPENMP
        GTEST_SKIP() << "Built without OpenMP, so without HPTT.";
#else
        if (process_thread_count() < 0) GTEST_SKIP() << "/proc/self/status is not available.";
        const Tensor source = random::uniform({64, 64, 64}, -1.0, 1.0, Device.cpu, /*seed=*/0);
        Tensor result;
        EXPECT_EQ(threads_started_by_transpose(source, /*openmp_threads=*/4, /*ncpus=*/1, result),
                  0);
        expect_reversed(result, source);
#endif
      }

      // A transpose too small to share out runs on the calling thread however many threads are
      // allowed: 16^3 Double elements are 32 KiB, below the 128 KiB at which a second thread starts
      // to pay.
      TEST(TensorTransposeThreads, SmallTransposeStartsNoThreads) {
#ifndef _OPENMP
        GTEST_SKIP() << "Built without OpenMP, so without HPTT.";
#else
        if (process_thread_count() < 0) GTEST_SKIP() << "/proc/self/status is not available.";
        const Tensor source = random::uniform({16, 16, 16}, -1.0, 1.0, Device.cpu, /*seed=*/0);
        Tensor result;
        EXPECT_EQ(threads_started_by_transpose(source, /*openmp_threads=*/4, /*ncpus=*/4, result),
                  0);
        expect_reversed(result, source);
#endif
      }

      // Cytnx's CPU kernels are single-threaded unless the user asks for more: Device.Ncpus is 1
      // unless CYTNX_NUM_THREADS is set.
      TEST(DeviceNcpus, DefaultIsOneUnlessSet) {
        if (std::getenv("CYTNX_NUM_THREADS") != nullptr)
          GTEST_SKIP() << "CYTNX_NUM_THREADS is set.";
        EXPECT_EQ(Device.Ncpus, 1);
      }

      // With the default setting, even a transpose with plenty of work runs on the calling thread
      // when OpenMP would allow more.
      TEST(TensorTransposeThreads, DefaultIsSingleThreaded) {
#ifndef _OPENMP
        GTEST_SKIP() << "Built without OpenMP, so without HPTT.";
#else
        if (std::getenv("CYTNX_NUM_THREADS") != nullptr)
          GTEST_SKIP() << "CYTNX_NUM_THREADS is set.";
        if (process_thread_count() < 0) GTEST_SKIP() << "/proc/self/status is not available.";
        const Tensor source = random::uniform({64, 64, 64}, -1.0, 1.0, Device.cpu, /*seed=*/0);
        Tensor result;
        EXPECT_EQ(threads_started_by_transpose(source, /*openmp_threads=*/4, Device.Ncpus, result),
                  0);
        expect_reversed(result, source);
#endif
      }

      // A transpose called from inside a user's OpenMP parallel region must copy the data on every
      // thread. HPTT skips the copy on a calling thread whose omp_get_thread_num() is not in its
      // thread-id list, which defaults to 0 .. numThreads - 1; with one thread allowed inside the
      // region, only thread 0 was in the list and thread 1 got an uninitialised result.
      TEST(TensorTransposeThreads, CopiesOnEveryThreadOfAParallelRegion) {
#ifndef _OPENMP
        GTEST_SKIP() << "Built without OpenMP, so without HPTT.";
#else
        const std::array<Tensor, 2> sources = {
          random::uniform({16, 16, 16}, -1.0, 1.0, Device.cpu, /*seed=*/10),
          random::uniform({16, 16, 16}, -1.0, 1.0, Device.cpu, /*seed=*/11)};
        std::array<Tensor, 2> results;
        std::array<int, 2> team_ids = {-1, -1};
        const int saved_threads = omp_get_max_threads();
        omp_set_num_threads(1);
  #pragma omp parallel num_threads(2)
        {
          const int thread = omp_get_thread_num();
          team_ids[thread] = thread;
          results[thread] = sources[thread].permute({2, 1, 0}).contiguous();
        }
        omp_set_num_threads(saved_threads);
        if (team_ids[1] != 1) GTEST_SKIP() << "OpenMP did not start a second thread.";
        for (std::size_t thread = 0; thread < 2; ++thread) {
          int wrong = 0;
          for (std::uint64_t i = 0; i < 16; ++i)
            for (std::uint64_t j = 0; j < 16; ++j)
              for (std::uint64_t k = 0; k < 16; ++k)
                if (results[thread].at<double>({i, j, k}) != sources[thread].at<double>({k, j, i}))
                  ++wrong;
          EXPECT_EQ(wrong, 0) << "thread " << thread;
        }
#endif
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
