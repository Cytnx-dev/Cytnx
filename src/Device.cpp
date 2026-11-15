#include "Device.hpp"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>

#include "cytnx_error.hpp"

namespace cytnx {

  namespace {

    // The default for Device.Ncpus: CYTNX_NUM_THREADS if it is set to a positive integer, otherwise
    // 1. Cytnx's CPU kernels are single-threaded unless the user asks for more: in DMRG workloads,
    // multi-threaded transposes interleaved with BLAS calls measured slower than single-threaded
    // ones (#939). This runs during static initialization, so a malformed value is reported on
    // stderr and ignored rather than thrown.
    int default_cpu_threads() {
      const char* setting = std::getenv("CYTNX_NUM_THREADS");
      if (setting == nullptr || *setting == '\0') return 1;
      char* end = nullptr;
      errno = 0;
      const long threads = std::strtol(setting, &end, 10);
      if (errno != 0 || *end != '\0' || threads < 1 || threads > INT_MAX) {
        std::fprintf(stderr,
                     "[WARNING] Cytnx: ignoring CYTNX_NUM_THREADS=\"%s\"; expected a positive "
                     "integer. Using 1 thread.\n",
                     setting);
        return 1;
      }
      return static_cast<int>(threads);
    }

  }  // namespace

  Device_class::Device_class() : Ngpus(0), Ncpus(default_cpu_threads()) {
#ifdef UNI_GPU

    // get all available gpus. A machine with no CUDA-capable device or no
    // NVIDIA driver installed legitimately returns cudaErrorNoDevice or
    // cudaErrorInsufficientDriver here -- the CUDA Runtime API is designed
    // to let a caller probe for GPU absence this way -- so treat those two
    // as Ngpus == 0 instead of routing them through checkCudaErrors' fatal
    // exit(), which would otherwise kill the process at import time on any
    // GPU-less machine running a CUDA-enabled build. Any other,
    // genuinely unexpected error still goes through checkCudaErrors.
    cudaError_t device_count_status = cudaGetDeviceCount(&Ngpus);
    if (device_count_status == cudaErrorNoDevice ||
        device_count_status == cudaErrorInsufficientDriver) {
      Ngpus = 0;
    } else {
      checkCudaErrors(device_count_status);
    }

    CanAccessPeer = std::vector<std::vector<bool>>(Ngpus, std::vector<bool>(Ngpus));
    // check can Peer Access, if can, open PCIE access to increase bandwidth
    //  Enable Peer Access when it's possible:
    //  https://stackoverflow.com/questions/31628041/how-to-copy-memory-between-different-gpus-in-cuda
    int cAP = 0;
    std::vector<int> isopen(Ngpus);
    for (int i = 0; i < Ngpus; i++) {
      CanAccessPeer[i][i] = 1;
      for (int j = i + 1; j < Ngpus; j++) {
        cudaDeviceCanAccessPeer(&cAP, i, j);
        if (cAP && !isopen[i]) {
          cudaDeviceEnablePeerAccess(i, 0);
          isopen[i] = 1;
        }
        if (cAP && !isopen[j]) {
          cudaDeviceEnablePeerAccess(j, 0);
          isopen[j] = 1;
        }
        if (cAP) {
          CanAccessPeer[i][j] = 1;
          CanAccessPeer[j][i] = 1;
        }
      }
    }
#endif  // UNI_GPU
  };

  Device_class::~Device_class(){

  };

  std::string Device_class::getname(const int& device_id) {
    if (device_id == this->cpu) {
      return std::string("cytnx device: CPU");
    } else if (device_id >= 0) {
      if (device_id >= Ngpus) {
        cytnx_error_msg(true, "%s", "[ERROR] invalid device_id, gpuid exceed limit");
        return std::string("");
      } else {
        return std::string("cytnx device: CUDA/GPU-id:") + std::to_string(device_id);
      }
    } else {
      cytnx_error_msg(true, "%s", "[ERROR] invalid device_id");
      return std::string("");
    }
  }
  void Device_class::Print_Property() {
    char* buffer = (char*)malloc(sizeof(char) * 256);
#ifdef UNI_GPU
    std::cout << "=== CUDA support ===" << std::endl;
    std::cout << ": Peer PCIE Access:" << std::endl;
    std::cout << "   ";
    for (int i = 0; i < this->Ngpus; i++) {
      sprintf(buffer, " %2d", i);
      std::cout << std::string(buffer);
    }
    std::cout << std::endl;

    std::cout << "   ";
    for (int i = 0; i < this->Ngpus; i++) {
      sprintf(buffer, "%s", "---");
      std::cout << std::string(buffer);
    }
    std::cout << std::endl;

    for (int i = 0; i < this->Ngpus; i++) {
      sprintf(buffer, "%2d|", i);
      std::cout << std::string(buffer);
      for (int j = 0; j < this->Ngpus; j++) {
        if (j == i) {
          sprintf(buffer, "%s", "  x");
          std::cout << std::string(buffer);
        } else {
          sprintf(buffer, "  %d", int(CanAccessPeer[i][j]));
          std::cout << std::string(buffer);
        }
      }
      std::cout << std::endl;
    }
    std::cout << "--------------------" << std::endl;
#else
    std::cout << "=== No CUDA support ===" << std::endl;
#endif

    free(buffer);
  }

  /*
  #ifdef UNI_GPU
  // See Device.cu
  #else
  void Device_class::cudaDeviceSynchronize(){
      cytnx_warning_msg(true,"[Warning] calling cudaDeviceSynchronize without CUDA support has no
  action%s","\n");
  }
  #endif
  */

  Device_class Device;
}  // namespace cytnx

// Maybe handle GPU part here.
