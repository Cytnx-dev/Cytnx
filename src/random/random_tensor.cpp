#include "random.hpp"

#include <algorithm>
#include <complex>
#include <limits>
#include <random>

#include "Type.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace random {

    Tensor random_tensor(const std::vector<cytnx_uint64>& shape, const double& low,
                         const double& high, const int& device, const unsigned int& seed,
                         const unsigned int& dtype) {
      if (dtype == Type.Float || dtype == Type.Double || dtype == Type.ComplexFloat ||
          dtype == Type.ComplexDouble) {
        return uniform(shape, low, high, device, seed, dtype);
      }

      Tensor out(shape, dtype, device);

      cytnx_uint64 total_elems = 1;
      for (const auto& dim : shape) {
        total_elems *= dim;
      }

      std::mt19937 gen(seed);

      Tensor cpu_tensor = out;
      if (device != cytnx::Device.cpu) {
        cpu_tensor = Tensor(shape, dtype, cytnx::Device.cpu);
      }

      switch (dtype) {
        case Type.Bool: {
          std::uniform_int_distribution<int> dist(0, 1);
          for (cytnx_uint64 i = 0; i < total_elems; ++i) {
            cpu_tensor.storage().at<cytnx_bool>(i) = static_cast<cytnx_bool>(dist(gen));
          }
          break;
        }

        case Type.Int16: {
          const int16_t low_int = static_cast<int16_t>(
            std::max(low, static_cast<double>(std::numeric_limits<int16_t>::min())));
          const int16_t high_int = static_cast<int16_t>(
            std::min(high, static_cast<double>(std::numeric_limits<int16_t>::max())));
          std::uniform_int_distribution<int16_t> dist(low_int, high_int);
          for (cytnx_uint64 i = 0; i < total_elems; ++i) {
            cpu_tensor.storage().at<cytnx_int16>(i) = dist(gen);
          }
          break;
        }

        case Type.Int32: {
          const int32_t low_int = static_cast<int32_t>(
            std::max(low, static_cast<double>(std::numeric_limits<int32_t>::min())));
          const int32_t high_int = static_cast<int32_t>(
            std::min(high, static_cast<double>(std::numeric_limits<int32_t>::max())));
          std::uniform_int_distribution<int32_t> dist(low_int, high_int);
          for (cytnx_uint64 i = 0; i < total_elems; ++i) {
            cpu_tensor.storage().at<cytnx_int32>(i) = dist(gen);
          }
          break;
        }

        case Type.Int64: {
          const int64_t low_int = static_cast<int64_t>(
            std::max(low, static_cast<double>(std::numeric_limits<int64_t>::min())));
          const int64_t high_int = static_cast<int64_t>(
            std::min(high, static_cast<double>(std::numeric_limits<int64_t>::max())));
          std::uniform_int_distribution<int64_t> dist(low_int, high_int);
          for (cytnx_uint64 i = 0; i < total_elems; ++i) {
            cpu_tensor.storage().at<cytnx_int64>(i) = dist(gen);
          }
          break;
        }

        case Type.Uint16: {
          const uint16_t low_uint = static_cast<uint16_t>(std::max(0.0, low));
          const uint16_t high_uint = static_cast<uint16_t>(
            std::min(high, static_cast<double>(std::numeric_limits<uint16_t>::max())));
          std::uniform_int_distribution<uint16_t> dist(low_uint, high_uint);
          for (cytnx_uint64 i = 0; i < total_elems; ++i) {
            cpu_tensor.storage().at<cytnx_uint16>(i) = dist(gen);
          }
          break;
        }

        case Type.Uint32: {
          const uint32_t low_uint = static_cast<uint32_t>(std::max(0.0, low));
          const uint32_t high_uint = static_cast<uint32_t>(
            std::min(high, static_cast<double>(std::numeric_limits<uint32_t>::max())));
          std::uniform_int_distribution<uint32_t> dist(low_uint, high_uint);
          for (cytnx_uint64 i = 0; i < total_elems; ++i) {
            cpu_tensor.storage().at<cytnx_uint32>(i) = dist(gen);
          }
          break;
        }

        case Type.Uint64: {
          const uint64_t low_uint = static_cast<uint64_t>(std::max(0.0, low));
          const uint64_t high_uint = static_cast<uint64_t>(
            std::min(high, static_cast<double>(std::numeric_limits<uint64_t>::max())));
          std::uniform_int_distribution<uint64_t> dist(low_uint, high_uint);
          for (cytnx_uint64 i = 0; i < total_elems; ++i) {
            cpu_tensor.storage().at<cytnx_uint64>(i) = dist(gen);
          }
          break;
        }

        default:
          cytnx_error_msg(true, "[random_tensor] Unsupported data type: %s",
                          Type.getname(dtype).c_str());
      }

      // Move to target device if needed
      if (device != cytnx::Device.cpu) {
        out = cpu_tensor.to(device);
      } else {
        out = cpu_tensor;
      }

      return out;
    }

  }  // namespace random
}  // namespace cytnx
