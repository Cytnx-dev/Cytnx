#include <stdlib.h>
#include <stdio.h>

#include <unordered_map>
#include <vector>
#include <cassert>

#include "cuQuantumQr_internal.hpp"

#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM
    #include <cuda_runtime.h>
    #include <cutensornet.h>

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

// int64_t computeCombinedExtent(const std::unordered_map<int32_t, int64_t> &extentMap,
//                               const std::vector<int32_t> &modes) {
//   int64_t combinedExtent{1};
//   for (auto mode : modes) {
//     auto it = extentMap.find(mode);
//     if (it != extentMap.end()) combinedExtent *= it->second;
//   }
//   return combinedExtent;
// }

namespace cytnx {
  namespace linalg_internal {

#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM

    void cuQuantumQr_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                                 boost::intrusive_ptr<Storage_base> &Q,
                                 boost::intrusive_ptr<Storage_base> &R,
                                 boost::intrusive_ptr<Storage_base> &D,
                                 boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
                                 const cytnx_int64 &N, const bool &is_d) {
      // const size_t cuTensornetVersion = cutensornetGetVersion();
      // printf("cuTensorNet-vers:%ld\n", cuTensornetVersion);

      /**********************************************
       * Tensor QR: T_{i,j,m,n} -> Q_{i,x,m} R_{n,x,j}
       ***********************************************/

      typedef float floatType;
      cudaDataType_t typeData = CUDA_C_64F;

      std::vector<int32_t> modesT{'j', 'i'};  // input
      std::vector<int32_t> modesQ{'s', 'i'};
      std::vector<int32_t> modesR{'j', 's'};  // QR output

      int n_tau = std::max(1, int(std::min(M, N)));

      std::vector<int64_t> extentT{N, M};
      std::vector<int64_t> extentQ{n_tau, M};
      std::vector<int64_t> extentR{N, n_tau};

      const int32_t numModesIn = modesT.size();
      const int32_t numModesQ = modesQ.size();
      const int32_t numModesR = modesR.size();

      void *D_T = in->Mem;
      void *D_Q = Q->Mem;
      void *D_R = R->Mem;

      /******************
       * cuTensorNet
       *******************/

      cudaStream_t stream;
      HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

      cutensornetHandle_t handle;
      HANDLE_ERROR(cutensornetCreate(&handle));

      /***************************
       * Create tensor descriptors
       ****************************/

      cutensornetTensorDescriptor_t descTensorIn;
      cutensornetTensorDescriptor_t descTensorQ;
      cutensornetTensorDescriptor_t descTensorR;

      const int64_t *strides = NULL;  // assuming fortran layout for all tensors

      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesIn, extentT.data(), strides,
                                                     modesT.data(), typeData, &descTensorIn));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesQ, extentQ.data(), strides,
                                                     modesQ.data(), typeData, &descTensorQ));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesR, extentR.data(), strides,
                                                     modesR.data(), typeData, &descTensorR));

      printf("Initialize the cuTensorNet library and create all tensor descriptors.\n");

      /********************************************
       * Query and allocate required workspace sizes
       *********************************************/

      cutensornetWorkspaceDescriptor_t workDesc;
      HANDLE_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));
      HANDLE_ERROR(cutensornetWorkspaceComputeQRSizes(handle, descTensorIn, descTensorQ,
                                                      descTensorR, workDesc));
      int64_t hostWorkspaceSize, deviceWorkspaceSize;

      // for tensor QR, it does not matter which cutensornetWorksizePref_t we pick
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
      /**********
       * Execution
       ***********/

      GPUTimer timer{stream};
      double minTimeCUTENSOR = 1e100;
      const int numRuns = 1;  // to get stable perf results
      for (int i = 0; i < numRuns; ++i) {
        // restore output
        // cudaMemsetAsync(D_Q, 0, sizeQ, stream);
        // cudaMemsetAsync(D_R, 0, sizeR, stream);
        cudaDeviceSynchronize();

        timer.start();
        HANDLE_ERROR(cutensornetTensorQR(handle, descTensorIn, D_T, descTensorQ, D_Q, descTensorR,
                                         D_R, workDesc, stream));
        // Synchronize and measure timing
        auto time = timer.seconds();
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
      }

      // printf("Performing QR\n");
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(Q, D_Q, sizeQ, cudaMemcpyDeviceToHost) );
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(R, D_R, sizeR, cudaMemcpyDeviceToHost) );

      cudaDeviceSynchronize();  // device synchronization.

      // printf("%.2f ms\n", minTimeCUTENSOR * 1000.f);
      /***************
       * Free resources
       ****************/

      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorIn));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorQ));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorR));
      HANDLE_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
      HANDLE_ERROR(cutensornetDestroy(handle));

      if (devWork) cudaFree(devWork);
      if (hostWork) free(hostWork);

      // printf("Free resource and exit.\n");
    }

    void cuQuantumQr_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                                 boost::intrusive_ptr<Storage_base> &Q,
                                 boost::intrusive_ptr<Storage_base> &R,
                                 boost::intrusive_ptr<Storage_base> &D,
                                 boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
                                 const cytnx_int64 &N, const bool &is_d) {
      // const size_t cuTensornetVersion = cutensornetGetVersion();
      // printf("cuTensorNet-vers:%ld\n", cuTensornetVersion);

      /**********************************************
       * Tensor QR: T_{i,j,m,n} -> Q_{i,x,m} R_{n,x,j}
       ***********************************************/

      typedef float floatType;
      cudaDataType_t typeData = CUDA_C_32F;

      std::vector<int32_t> modesT{'j', 'i'};  // input
      std::vector<int32_t> modesQ{'s', 'i'};
      std::vector<int32_t> modesR{'j', 's'};  // QR output

      int n_tau = std::max(1, int(std::min(M, N)));

      std::vector<int64_t> extentT{N, M};
      std::vector<int64_t> extentQ{n_tau, M};
      std::vector<int64_t> extentR{N, n_tau};

      const int32_t numModesIn = modesT.size();
      const int32_t numModesQ = modesQ.size();
      const int32_t numModesR = modesR.size();

      void *D_T = in->Mem;
      void *D_Q = Q->Mem;
      void *D_R = R->Mem;

      /******************
       * cuTensorNet
       *******************/

      cudaStream_t stream;
      HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

      cutensornetHandle_t handle;
      HANDLE_ERROR(cutensornetCreate(&handle));

      /***************************
       * Create tensor descriptors
       ****************************/

      cutensornetTensorDescriptor_t descTensorIn;
      cutensornetTensorDescriptor_t descTensorQ;
      cutensornetTensorDescriptor_t descTensorR;

      const int64_t *strides = NULL;  // assuming fortran layout for all tensors

      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesIn, extentT.data(), strides,
                                                     modesT.data(), typeData, &descTensorIn));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesQ, extentQ.data(), strides,
                                                     modesQ.data(), typeData, &descTensorQ));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesR, extentR.data(), strides,
                                                     modesR.data(), typeData, &descTensorR));

      printf("Initialize the cuTensorNet library and create all tensor descriptors.\n");

      /********************************************
       * Query and allocate required workspace sizes
       *********************************************/

      cutensornetWorkspaceDescriptor_t workDesc;
      HANDLE_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));
      HANDLE_ERROR(cutensornetWorkspaceComputeQRSizes(handle, descTensorIn, descTensorQ,
                                                      descTensorR, workDesc));
      int64_t hostWorkspaceSize, deviceWorkspaceSize;

      // for tensor QR, it does not matter which cutensornetWorksizePref_t we pick
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
      /**********
       * Execution
       ***********/

      GPUTimer timer{stream};
      double minTimeCUTENSOR = 1e100;
      const int numRuns = 1;  // to get stable perf results
      for (int i = 0; i < numRuns; ++i) {
        // restore output
        // cudaMemsetAsync(D_Q, 0, sizeQ, stream);
        // cudaMemsetAsync(D_R, 0, sizeR, stream);
        cudaDeviceSynchronize();

        timer.start();
        HANDLE_ERROR(cutensornetTensorQR(handle, descTensorIn, D_T, descTensorQ, D_Q, descTensorR,
                                         D_R, workDesc, stream));
        // Synchronize and measure timing
        auto time = timer.seconds();
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
      }

      // printf("Performing QR\n");
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(Q, D_Q, sizeQ, cudaMemcpyDeviceToHost) );
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(R, D_R, sizeR, cudaMemcpyDeviceToHost) );

      cudaDeviceSynchronize();  // device synchronization.

      // printf("%.2f ms\n", minTimeCUTENSOR * 1000.f);
      /***************
       * Free resources
       ****************/

      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorIn));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorQ));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorR));
      HANDLE_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
      HANDLE_ERROR(cutensornetDestroy(handle));

      if (devWork) cudaFree(devWork);
      if (hostWork) free(hostWork);

      // printf("Free resource and exit.\n");
    }
    void cuQuantumQr_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                                boost::intrusive_ptr<Storage_base> &Q,
                                boost::intrusive_ptr<Storage_base> &R,
                                boost::intrusive_ptr<Storage_base> &D,
                                boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
                                const cytnx_int64 &N, const bool &is_d) {
      // const size_t cuTensornetVersion = cutensornetGetVersion();
      // printf("cuTensorNet-vers:%ld\n", cuTensornetVersion);

      /**********************************************
       * Tensor QR: T_{i,j,m,n} -> Q_{i,x,m} R_{n,x,j}
       ***********************************************/

      typedef float floatType;
      cudaDataType_t typeData = CUDA_R_64F;

      std::vector<int32_t> modesT{'j', 'i'};  // input
      std::vector<int32_t> modesQ{'s', 'i'};
      std::vector<int32_t> modesR{'j', 's'};  // QR output

      int n_tau = std::max(1, int(std::min(M, N)));

      std::vector<int64_t> extentT{N, M};
      std::vector<int64_t> extentQ{n_tau, M};
      std::vector<int64_t> extentR{N, n_tau};

      const int32_t numModesIn = modesT.size();
      const int32_t numModesQ = modesQ.size();
      const int32_t numModesR = modesR.size();

      void *D_T = in->Mem;
      void *D_Q = Q->Mem;
      void *D_R = R->Mem;

      /******************
       * cuTensorNet
       *******************/

      cudaStream_t stream;
      HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

      cutensornetHandle_t handle;
      HANDLE_ERROR(cutensornetCreate(&handle));

      /***************************
       * Create tensor descriptors
       ****************************/

      cutensornetTensorDescriptor_t descTensorIn;
      cutensornetTensorDescriptor_t descTensorQ;
      cutensornetTensorDescriptor_t descTensorR;

      const int64_t *strides = NULL;  // assuming fortran layout for all tensors

      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesIn, extentT.data(), strides,
                                                     modesT.data(), typeData, &descTensorIn));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesQ, extentQ.data(), strides,
                                                     modesQ.data(), typeData, &descTensorQ));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesR, extentR.data(), strides,
                                                     modesR.data(), typeData, &descTensorR));

      printf("Initialize the cuTensorNet library and create all tensor descriptors.\n");

      /********************************************
       * Query and allocate required workspace sizes
       *********************************************/

      cutensornetWorkspaceDescriptor_t workDesc;
      HANDLE_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));
      HANDLE_ERROR(cutensornetWorkspaceComputeQRSizes(handle, descTensorIn, descTensorQ,
                                                      descTensorR, workDesc));
      int64_t hostWorkspaceSize, deviceWorkspaceSize;

      // for tensor QR, it does not matter which cutensornetWorksizePref_t we pick
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
      /**********
       * Execution
       ***********/

      GPUTimer timer{stream};
      double minTimeCUTENSOR = 1e100;
      const int numRuns = 1;  // to get stable perf results
      for (int i = 0; i < numRuns; ++i) {
        // restore output
        // cudaMemsetAsync(D_Q, 0, sizeQ, stream);
        // cudaMemsetAsync(D_R, 0, sizeR, stream);
        cudaDeviceSynchronize();

        timer.start();
        HANDLE_ERROR(cutensornetTensorQR(handle, descTensorIn, D_T, descTensorQ, D_Q, descTensorR,
                                         D_R, workDesc, stream));
        // Synchronize and measure timing
        auto time = timer.seconds();
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
      }

      // printf("Performing QR\n");
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(Q, D_Q, sizeQ, cudaMemcpyDeviceToHost) );
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(R, D_R, sizeR, cudaMemcpyDeviceToHost) );

      cudaDeviceSynchronize();  // device synchronization.

      // printf("%.2f ms\n", minTimeCUTENSOR * 1000.f);
      /***************
       * Free resources
       ****************/

      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorIn));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorQ));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorR));
      HANDLE_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
      HANDLE_ERROR(cutensornetDestroy(handle));

      if (devWork) cudaFree(devWork);
      if (hostWork) free(hostWork);

      // printf("Free resource and exit.\n");
    }
    void cuQuantumQr_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                                boost::intrusive_ptr<Storage_base> &Q,
                                boost::intrusive_ptr<Storage_base> &R,
                                boost::intrusive_ptr<Storage_base> &D,
                                boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
                                const cytnx_int64 &N, const bool &is_d) {
      // const size_t cuTensornetVersion = cutensornetGetVersion();
      // printf("cuTensorNet-vers:%ld\n", cuTensornetVersion);

      /**********************************************
       * Tensor QR: T_{i,j,m,n} -> Q_{i,x,m} R_{n,x,j}
       ***********************************************/

      typedef float floatType;
      cudaDataType_t typeData = CUDA_R_32F;

      std::vector<int32_t> modesT{'j', 'i'};  // input
      std::vector<int32_t> modesQ{'s', 'i'};
      std::vector<int32_t> modesR{'j', 's'};  // QR output

      int n_tau = std::max(1, int(std::min(M, N)));

      std::vector<int64_t> extentT{N, M};
      std::vector<int64_t> extentQ{n_tau, M};
      std::vector<int64_t> extentR{N, n_tau};

      const int32_t numModesIn = modesT.size();
      const int32_t numModesQ = modesQ.size();
      const int32_t numModesR = modesR.size();

      void *D_T = in->Mem;
      void *D_Q = Q->Mem;
      void *D_R = R->Mem;

      /******************
       * cuTensorNet
       *******************/

      cudaStream_t stream;
      HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

      cutensornetHandle_t handle;
      HANDLE_ERROR(cutensornetCreate(&handle));

      /***************************
       * Create tensor descriptors
       ****************************/

      cutensornetTensorDescriptor_t descTensorIn;
      cutensornetTensorDescriptor_t descTensorQ;
      cutensornetTensorDescriptor_t descTensorR;

      const int64_t *strides = NULL;  // assuming fortran layout for all tensors

      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesIn, extentT.data(), strides,
                                                     modesT.data(), typeData, &descTensorIn));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesQ, extentQ.data(), strides,
                                                     modesQ.data(), typeData, &descTensorQ));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesR, extentR.data(), strides,
                                                     modesR.data(), typeData, &descTensorR));

      printf("Initialize the cuTensorNet library and create all tensor descriptors.\n");

      /********************************************
       * Query and allocate required workspace sizes
       *********************************************/

      cutensornetWorkspaceDescriptor_t workDesc;
      HANDLE_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));
      HANDLE_ERROR(cutensornetWorkspaceComputeQRSizes(handle, descTensorIn, descTensorQ,
                                                      descTensorR, workDesc));
      int64_t hostWorkspaceSize, deviceWorkspaceSize;

      // for tensor QR, it does not matter which cutensornetWorksizePref_t we pick
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
      /**********
       * Execution
       ***********/

      GPUTimer timer{stream};
      double minTimeCUTENSOR = 1e100;
      const int numRuns = 1;  // to get stable perf results
      for (int i = 0; i < numRuns; ++i) {
        // restore output
        // cudaMemsetAsync(D_Q, 0, sizeQ, stream);
        // cudaMemsetAsync(D_R, 0, sizeR, stream);
        cudaDeviceSynchronize();

        timer.start();
        HANDLE_ERROR(cutensornetTensorQR(handle, descTensorIn, D_T, descTensorQ, D_Q, descTensorR,
                                         D_R, workDesc, stream));
        // Synchronize and measure timing
        auto time = timer.seconds();
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
      }

      // printf("Performing QR\n");
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(Q, D_Q, sizeQ, cudaMemcpyDeviceToHost) );
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(R, D_R, sizeR, cudaMemcpyDeviceToHost) );

      cudaDeviceSynchronize();  // device synchronization.

      // printf("%.2f ms\n", minTimeCUTENSOR * 1000.f);
      /***************
       * Free resources
       ****************/

      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorIn));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorQ));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorR));
      HANDLE_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
      HANDLE_ERROR(cutensornetDestroy(handle));

      if (devWork) cudaFree(devWork);
      if (hostWork) free(hostWork);

      // printf("Free resource and exit.\n");
    }
  #endif
#endif
  }  // namespace linalg_internal
}  // namespace cytnx
