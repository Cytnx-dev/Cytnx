#include <stdlib.h>
#include <stdio.h>

#include <unordered_map>
#include <vector>
#include <cassert>

#include "cuQuantumGeSvd_internal.hpp"

#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM
    #include <cuda_runtime.h>
    #include <cutensornet.h>

  //  #define HANDLE_ERROR(x)                                           \
  //  { const auto err = x;                                             \
  //  if( err != CUTENSORNET_STATUS_SUCCESS )                           \
  //  { printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__); return err; } \
  //  };

  //  #define HANDLE_CUDA_ERROR(x)                                      \
  //  {  const auto err = x;                                            \
  //     if( err != cudaSuccess )                                       \
  //     { printf("Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); return err; } \
  //  };
    #define HANDLE_ERROR(x)                                                           \
      {                                                                               \
        const auto err = x;                                                           \
        if (err != CUTENSORNET_STATUS_SUCCESS) {                                      \
          printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__); \
          fflush(stdout);                                                             \
        }                                                                             \
      };

    #define HANDLE_CUDA_ERROR(x)                                                    \
      {                                                                             \
        const auto err = x;                                                         \
        if (err != cudaSuccess) {                                                   \
          printf("CUDA Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); \
          fflush(stdout);                                                           \
        }                                                                           \
      };

struct GPUTimer {
  GPUTimer(cudaStream_t stream) : stream_(stream) {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~GPUTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() { cudaEventRecord(start_, stream_); }

  float seconds() {
    cudaEventRecord(stop_, stream_);
    cudaEventSynchronize(stop_);
    float time;
    cudaEventElapsedTime(&time, start_, stop_);
    return time * 1e-3;
  }

 private:
  cudaEvent_t start_, stop_;
  cudaStream_t stream_;
};

  #endif
#endif

int64_t computeCombinedExtent(const std::unordered_map<int32_t, int64_t> &extentMap,
                              const std::vector<int32_t> &modes) {
  int64_t combinedExtent{1};
  for (auto mode : modes) {
    auto it = extentMap.find(mode);
    if (it != extentMap.end()) combinedExtent *= it->second;
  }
  return combinedExtent;
}

namespace cytnx {
  namespace linalg_internal {

#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM
    /// cuGeSvd
    void cuQuantumGeSvd_internal_cd(const Tensor &Tin, const cytnx_uint64 &keepdim,
                                    const double &err, const unsigned int &return_err, Tensor U,
                                    Tensor S, Tensor vT) {
      const size_t cuTensornetVersion = cutensornetGetVersion();
      printf("cuTensorNet-vers:%ld\n", cuTensornetVersion);

      cudaDeviceProp prop;
      int deviceId = Tin.device();
      HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
      HANDLE_CUDA_ERROR(cudaGetDevice(&deviceId));
      HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceId));

      printf("===== device info ======\n");
      printf("GPU-name:%s\n", prop.name);
      printf("GPU-clock:%d\n", prop.clockRate);
      printf("GPU-memoryClock:%d\n", prop.memoryClockRate);
      printf("GPU-nSM:%d\n", prop.multiProcessorCount);
      printf("GPU-major:%d\n", prop.major);
      printf("GPU-minor:%d\n", prop.minor);
      printf("========================\n");

      typedef float floatType;
      cudaDataType_t typeData = CUDA_C_64F;
      /******************
       * cuTensorNet
       *******************/

      cudaStream_t stream;
      HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

      cutensornetHandle_t handle;
      HANDLE_ERROR(cutensornetCreate(&handle));

      /**************************
       * Create tensor descriptors
       ***************************/
      std::vector<int32_t> modesT{'j', 'i'};
      std::vector<int32_t> modesU{'s', 'i'};
      std::vector<int32_t> modesV{'j', 's'};

      std::vector<int64_t> extentT{(int64_t)Tin.shape()[1], (int64_t)Tin.shape()[0]};
      std::vector<int64_t> extentU{(int64_t)U.shape()[1], (int64_t)U.shape()[0]};
      std::vector<int64_t> extentV{(int64_t)vT.shape()[1], (int64_t)vT.shape()[0]};

      const int32_t numModesIn = modesT.size();
      const int32_t numModesU = modesU.size();
      const int32_t numModesV = modesV.size();

      void *D_T = Tin._impl->storage()._impl->Mem;
      void *D_U = U._impl->storage()._impl->Mem;
      void *D_S = S._impl->storage()._impl->Mem;
      void *D_V = vT._impl->storage()._impl->Mem;

      cutensornetTensorDescriptor_t descTensorIn;
      cutensornetTensorDescriptor_t descTensorU;
      cutensornetTensorDescriptor_t descTensorV;

      const int64_t *strides = NULL;  // assuming fortran layout for all tensors

      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesIn, extentT.data(), strides,
                                                     modesT.data(), typeData, &descTensorIn));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesU, extentU.data(), strides,
                                                     modesU.data(), typeData, &descTensorU));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesV, extentV.data(), strides,
                                                     modesV.data(), typeData, &descTensorV));

      printf("Initialize the cuTensorNet library and create all tensor descriptors.\n");

      /**********************************************
       * Setup SVD algorithm and truncation parameters
       ***********************************************/

      cutensornetTensorSVDConfig_t svdConfig;
      HANDLE_ERROR(cutensornetCreateTensorSVDConfig(handle, &svdConfig));

      // set up truncation parameters
      double absCutoff = err;
      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(handle, svdConfig,
                                                          CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF,
                                                          &absCutoff, sizeof(absCutoff)));
      double relCutoff = 0;
      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(handle, svdConfig,
                                                          CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF,
                                                          &relCutoff, sizeof(relCutoff)));

      // optional: choose gesvdj algorithm with customized parameters. Default is gesvd.
      cutensornetTensorSVDAlgo_t svdAlgo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ;
      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(
        handle, svdConfig, CUTENSORNET_TENSOR_SVD_CONFIG_ALGO, &svdAlgo, sizeof(svdAlgo)));
      cutensornetGesvdjParams_t gesvdjParams{/*tol=*/0, /*maxSweeps=*/80};
      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(handle, svdConfig,
                                                          CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS,
                                                          &gesvdjParams, sizeof(gesvdjParams)));
      printf("Set up SVDConfig to use GESVDJ algorithm with truncation\n");

      /********************************************************
       * Create SVDInfo to record runtime SVD truncation details
       *********************************************************/

      cutensornetTensorSVDInfo_t svdInfo;
      HANDLE_ERROR(cutensornetCreateTensorSVDInfo(handle, &svdInfo));

      // Sphinx: #6
      /**************************************************************
       * Query the required workspace sizes and allocate memory
       **************************************************************/

      cutensornetWorkspaceDescriptor_t workDesc;
      HANDLE_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));
      HANDLE_ERROR(cutensornetWorkspaceComputeSVDSizes(handle, descTensorIn, descTensorU,
                                                       descTensorV, svdConfig, workDesc));
      int64_t hostWorkspaceSize, deviceWorkspaceSize;
      // for tensor SVD, it does not matter which cutensornetWorksizePref_t we pick
      HANDLE_ERROR(cutensornetWorkspaceGetMemorySize(
        handle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
        CUTENSORNET_WORKSPACE_SCRATCH, &deviceWorkspaceSize));
      HANDLE_ERROR(cutensornetWorkspaceGetMemorySize(
        handle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_HOST,
        CUTENSORNET_WORKSPACE_SCRATCH, &hostWorkspaceSize));

      void *devWork = nullptr, *hostWork = nullptr;
      if (deviceWorkspaceSize > 0) {
        HANDLE_CUDA_ERROR(cudaMalloc(&devWork, deviceWorkspaceSize));
      }
      if (hostWorkspaceSize > 0) {
        hostWork = malloc(hostWorkspaceSize);
      }
      HANDLE_ERROR(cutensornetWorkspaceSetMemory(handle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                                                 CUTENSORNET_WORKSPACE_SCRATCH, devWork,
                                                 deviceWorkspaceSize));
      HANDLE_ERROR(cutensornetWorkspaceSetMemory(handle, workDesc, CUTENSORNET_MEMSPACE_HOST,
                                                 CUTENSORNET_WORKSPACE_SCRATCH, hostWork,
                                                 hostWorkspaceSize));

      // Sphinx: #7
      /**********
       * Execution
       ***********/

      GPUTimer timer{stream};
      double minTimeCUTENSOR = 1e100;
      const int numRuns = 1;  // to get stable perf results
      for (int i = 0; i < numRuns; ++i) {
        // // restore output
        // cudaMemsetAsync(D_U, 0, sizeU, stream);
        // cudaMemsetAsync(D_S, 0, sizeS, stream);
        // cudaMemsetAsync(D_V, 0, sizeV, stream);
        // cudaDeviceSynchronize();

        // With value-based truncation, `cutensornetTensorSVD` can potentially update the shared
        // extent in descTensorU/V. We here restore descTensorU/V to the original problem.
        HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorU));
        HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorV));
        HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesU, extentU.data(), strides,
                                                       modesU.data(), typeData, &descTensorU));
        HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesV, extentV.data(), strides,
                                                       modesV.data(), typeData, &descTensorV));

        timer.start();
        HANDLE_ERROR(cutensornetTensorSVD(handle, descTensorIn, D_T, descTensorU, D_U, D_S,
                                          descTensorV, D_V, svdConfig, svdInfo, workDesc, stream));
        // Synchronize and measure timing
        auto time = timer.seconds();
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
      }

      printf("Performing SVD\n");

      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(U, D_U, sizeU, cudaMemcpyDeviceToHost) );
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(S, D_S, sizeS, cudaMemcpyDeviceToHost) );
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(V, D_V, sizeV, cudaMemcpyDeviceToHost) );

      // Sphinx: #8
      /*************************************
       * Query runtime truncation information
       **************************************/

      double discardedWeight{0};
      int64_t reducedExtent{0};
      cutensornetGesvdjStatus_t gesvdjStatus;
      cudaDeviceSynchronize();  // device synchronization.
      HANDLE_ERROR(cutensornetTensorSVDInfoGetAttribute(
        handle, svdInfo, CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT, &discardedWeight,
        sizeof(discardedWeight)));
      HANDLE_ERROR(cutensornetTensorSVDInfoGetAttribute(handle, svdInfo,
                                                        CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT,
                                                        &reducedExtent, sizeof(reducedExtent)));
      HANDLE_ERROR(cutensornetTensorSVDInfoGetAttribute(handle, svdInfo,
                                                        CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS,
                                                        &gesvdjStatus, sizeof(gesvdjStatus)));

      printf("elapsed time: %.2f ms\n", minTimeCUTENSOR * 1000.f);
      printf("GESVDJ residual: %.4f, runtime sweeps = %d\n", gesvdjStatus.residual,
             gesvdjStatus.sweeps);
      printf("reduced extent found at runtime: %lu\n", reducedExtent);
      printf("discarded weight: %.2f\n", discardedWeight);

      // Sphinx: #9
      /***************
       * Free resources
       ****************/

      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorIn));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorU));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorV));
      HANDLE_ERROR(cutensornetDestroyTensorSVDConfig(svdConfig));
      HANDLE_ERROR(cutensornetDestroyTensorSVDInfo(svdInfo));
      HANDLE_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
      HANDLE_ERROR(cutensornetDestroy(handle));

      // std::cout<<S;
      // std::cout<<U;
      // std::cout<<vT;
      // if (T) free(T);
      // if (U) free(U);
      // if (S) free(S);
      // if (V) free(V);
      // if (D_T) cudaFree(D_T);
      // if (D_U) cudaFree(D_U);
      // if (D_S) cudaFree(D_S);
      // if (D_V) cudaFree(D_V);
      if (devWork) cudaFree(devWork);
      if (hostWork) free(hostWork);

      printf("Free resource and exit.\n");
    }
  #endif
#endif
  }  // namespace linalg_internal
}  // namespace cytnx
