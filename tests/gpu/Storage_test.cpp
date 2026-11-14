#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "cytnx.hpp"

namespace cytnx {
  namespace {

    // State shared with a host callback that blocks the default stream until the test releases it,
    // or until a timeout passes.
    struct StreamBlocker {
      std::atomic<bool> release{false};
      std::atomic<bool> timed_out{false};
    };

    void CUDART_CB block_until_released(void* data) {
      auto* blocker = static_cast<StreamBlocker*>(data);
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (!blocker->release.load()) {
        if (std::chrono::steady_clock::now() > deadline) {
          blocker->timed_out = true;
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    // Typed access to CPU storage must not wait for pending GPU work (#1260). The default stream
    // is blocked by a host callback that only the test releases, after the accesses. If an
    // accessor called cudaDeviceSynchronize, it would wait until the callback timed out.
    TEST(StorageGpu, CpuTypedAccessDoesNotWaitForDevice) {
      Tensor t = arange(0, 4, 1, Type.Double, Device.cpu);
      Storage storage = t.storage();
      StreamBlocker blocker;
      ASSERT_EQ(cudaLaunchHostFunc(0, block_until_released, &blocker), cudaSuccess);

      const double via_tensor_at = t.at<double>({2});
      const double via_storage_at = storage.at<double>(1);
      const double via_back = storage.back<double>();
      const double via_data = storage.data<double>()[0];

      blocker.release = true;
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      EXPECT_FALSE(blocker.timed_out) << "a CPU storage accessor synchronized the device";
      EXPECT_EQ(via_tensor_at, 2.0);
      EXPECT_EQ(via_storage_at, 1.0);
      EXPECT_EQ(via_back, 3.0);
      EXPECT_EQ(via_data, 0.0);
    }

  }  // namespace
}  // namespace cytnx
