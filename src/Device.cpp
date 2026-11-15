#include "Device.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <string>

#ifdef __linux__
  #include <sched.h>
#endif
#ifdef _OPENMP
  #include <omp.h>
#endif

#include "cytnx_error.hpp"

namespace cytnx {

  namespace {

    // Physical cores among the CPUs this process may run on, counting hyperthread siblings once,
    // or 0 where the topology cannot be read.
    int physical_cores_in_affinity() {
#ifdef __linux__
      cpu_set_t allowed;
      CPU_ZERO(&allowed);
      if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return 0;
      std::set<std::string> cores;  // one sibling list per physical core
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        const std::string topology =
          "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        std::ifstream siblings(topology + "core_cpus_list");
        if (!siblings.is_open()) siblings.open(topology + "thread_siblings_list");
        std::string list;
        if (!std::getline(siblings, list)) return 0;
        cores.insert(list);
      }
      return static_cast<int>(cores.size());
#else
      return 0;
#endif
    }

    // The default for Device.Ncpus. OpenMP's thread count honours OMP_NUM_THREADS and the
    // process's CPU affinity (taskset, cpusets, batch-system allocations such as Slurm's), unlike
    // std::thread::hardware_concurrency(). It is capped at the physical cores, because thread teams
    // that also use hyperthread siblings make Cytnx's transposes slower and erratic (#939). Without
    // OpenMP, Cytnx's CPU kernels are single-threaded.
    int default_cpu_threads() {
#ifdef _OPENMP
      const int openmp_threads = omp_get_max_threads();
      const int cores = physical_cores_in_affinity();
      return cores > 0 ? std::min(openmp_threads, cores) : openmp_threads;
#else
      return 1;
#endif
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
