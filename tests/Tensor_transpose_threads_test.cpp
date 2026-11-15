#include <omp.h>

#include <algorithm>
#include <complex>
#include <iostream>
#include <string>
#include <type_traits>

#include "cytnx.hpp"

namespace {
  thread_local int expected_team = 0;
  thread_local int observed_regions = 0;
  thread_local int wrong_teams = 0;

#ifdef CYTNX_TEST_GOMP_TEAMS
  struct ObservedCall {
    void (*function)(void*);
    void* data;
    int expected;
    int* failures;
  };

  void observe_team(void* pointer) {
    auto& call = *static_cast<ObservedCall*>(pointer);
    if (omp_get_thread_num() == 0 && omp_get_num_threads() != call.expected) {
      ++*call.failures;
    }
    call.function(call.data);
  }
#endif
}  // namespace

#ifdef CYTNX_TEST_GOMP_TEAMS
// Observe actual HPTT teams, without relying on idle OS threads or a getter.
extern "C" void real_gomp_parallel(void (*function)(void*), void* data, unsigned threads,
                                   unsigned flags) asm("__real_GOMP_parallel");
extern "C" void wrap_gomp_parallel(void (*function)(void*), void* data, unsigned threads,
                                   unsigned flags) asm("__wrap_GOMP_parallel");
extern "C" void wrap_gomp_parallel(void (*function)(void*), void* data, unsigned threads,
                                   unsigned flags) {
  if (expected_team == 0) {
    real_gomp_parallel(function, data, threads, flags);
    return;
  }
  ++observed_regions;
  ObservedCall call{function, data, expected_team, &wrong_teams};
  real_gomp_parallel(observe_team, &call, threads, flags);
}
#endif

namespace {
  template <typename T>
  T sample(int index) {
    const double real = (index % 31 - 15) * 0.25;
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>) {
      using Real = typename T::value_type;
      return T(static_cast<Real>(real), static_cast<Real>((index % 13 - 6) * 0.5));
    } else {
      return static_cast<T>(real);
    }
  }

  template <typename T>
  int check_transpose(unsigned dtype, int threads, bool inplace) {
    cytnx::Tensor source({5, 7, 9}, dtype);
    auto source_storage = source.storage();
    for (int i = 0; i < 315; ++i) source_storage.at<T>(i) = sample<T>(i);
    auto result = source.permute({2, 0, 1});
    expected_team = threads;
    observed_regions = 0;
    wrong_teams = 0;
    if (inplace) {
      result.contiguous_();
    } else {
      result = result.contiguous();
    }
    expected_team = 0;
    int failures = 0;
#ifdef CYTNX_TEST_GOMP_TEAMS
    failures += wrong_teams + (threads > 1 && observed_regions == 0);
#endif
    failures += result.dtype() != dtype || !result.is_contiguous();
    const auto shape = result.shape();
    failures += shape.size() != 3 || shape[0] != 9 || shape[1] != 5 || shape[2] != 7;
    const auto output = result.storage();
    // Independent row-major coordinate oracle, including every element and its dtype.
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 7; ++j) {
        for (int k = 0; k < 9; ++k) {
          const int source_index = (i * 7 + j) * 9 + k;
          const int target_index = (k * 5 + i) * 7 + j;
          failures += output.at<T>(target_index) != sample<T>(source_index);
          failures += source_storage.at<T>(source_index) != sample<T>(source_index);
        }
      }
    }
    return failures;
  }

  int check_types(int threads) {
    int failures = 0;
    for (bool inplace : {false, true}) {
      failures += check_transpose<float>(cytnx::Type.Float, threads, inplace);
      failures += check_transpose<double>(cytnx::Type.Double, threads, inplace);
      failures += check_transpose<std::complex<float>>(cytnx::Type.ComplexFloat, threads, inplace);
      failures +=
        check_transpose<std::complex<double>>(cytnx::Type.ComplexDouble, threads, inplace);
    }
    return failures;
  }
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "invalid") {
    auto source = cytnx::arange(315).reshape({5, 7, 9});
    try {
      auto result = source.permute({2, 0, 1}).contiguous();
    } catch (const cytnx::error& error) {
      return std::string(error.what()).find("HPTT_NUM_THREADS") == std::string::npos;
    }
    std::cerr << "Invalid HPTT_NUM_THREADS was accepted\n";
    return 1;
  }
  if (argc != 4) return 1;
  const int initial_count = std::stoi(argv[1]);
  const int inner_count = std::stoi(argv[2]);
  const int override_count = std::stoi(argv[3]);
  omp_set_dynamic(0);
  int failures = check_types(initial_count);
  int outer_size = 0;
#pragma omp parallel num_threads(4) reduction(+ : failures)
  {
#pragma omp single
    outer_size = omp_get_num_threads();
    // A nonzero outer worker must complete the entire transpose even when
    // the nested list requests one thread or a thread limit serializes it.
    if (omp_get_thread_num() == 3) failures += check_types(inner_count);
  }
  failures += outer_size != 4;
  omp_set_num_threads(2);
  failures += check_types(override_count > 0 ? override_count : 2);
  omp_set_num_threads(1);
  failures += check_types(override_count > 0 ? override_count : 1);
  if (failures) std::cerr << failures << " transpose or team-size checks failed\n";
  return failures ? 1 : 0;
}
