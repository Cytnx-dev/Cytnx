#ifdef UNI_CUQUANTUM
#include <stdlib.h>
#include <stdio.h>

#include <unordered_map>
#include <vector>
#include <cassert>

#include <cuda_runtime.h>
#include <cutensornet.h>

#include <cytnx.hpp>
#include "utils/cutensornet.hpp"

namespace cytnx {

#ifdef UNI_GPU

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

  // Call cutensornet
  void cuTensornet_(const int32_t numInputs, void* R_d, int32_t nmodeR, int64_t* extentR,
                    int32_t* modesR, void* rawDataIn_d[], int32_t* modesIn[],
                    int32_t const numModesIn[], int64_t* extentsIn[], int64_t* stridesIn[],
                    bool verbose) {
    static_assert(sizeof(size_t) == sizeof(int64_t),
                  "Please build this sample on a 64-bit architecture!");

    // Check cuTensorNet version
    const size_t cuTensornetVersion = cutensornetGetVersion();
    if (verbose) printf("cuTensorNet version: %ld\n", cuTensornetVersion);

    // Set GPU device
    int numDevices{0};
    HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices));
    const int deviceId = 0;
    HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
    cudaDeviceProp prop;
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceId));

    if (verbose) {
      printf("===== device info ======\n");
      printf("GPU-name:%s\n", prop.name);
      printf("GPU-clock:%d\n", prop.clockRate);
      printf("GPU-memoryClock:%d\n", prop.memoryClockRate);
      printf("GPU-nSM:%d\n", prop.multiProcessorCount);
      printf("GPU-major:%d\n", prop.major);
      printf("GPU-minor:%d\n", prop.minor);
      printf("========================\n");
    }

    typedef float floatType;
    cudaDataType_t typeData = CUDA_R_32F;
    cutensornetComputeType_t typeCompute = CUTENSORNET_COMPUTE_32F;

    if (verbose) printf("Included headers and defined data types\n");

    /**********************
     * Computing: R_{k,l} = A_{a,b,c,d,e,f} B_{b,g,h,e,i,j} C_{m,a,g,f,i,k} D_{l,c,h,d,j,m}
     **********************/

    if (verbose) printf("Defined tensor network, modes, and extents\n");

    // Sphinx: #3
    /**********************
     * Allocating data
     **********************/

    // if(verbose)
    //    printf("Total GPU memory used for tensor storage: %.2f GiB\n",
    //           (sizeA + sizeB + sizeC + sizeD + sizeR) / 1024. /1024. / 1024);

    if (verbose) printf("Allocated GPU memory for data, and initialize data\n");

    // Sphinx: #4
    /*************************
     * cuTensorNet
     *************************/

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cutensornetHandle_t handle;
    HANDLE_ERROR(cutensornetCreate(&handle));

    /*******************************
     * Create Network Descriptor
     *******************************/

    // Set up tensor network
    cutensornetNetworkDescriptor_t descNet;
    HANDLE_ERROR(cutensornetCreateNetworkDescriptor(
      handle, numInputs, numModesIn, extentsIn, stridesIn, modesIn, NULL, nmodeR, extentR,
      /*stridesOut = */ NULL, modesR, typeData, typeCompute, &descNet));

    if (verbose)
      printf("Initialized the cuTensorNet library and created a tensor network descriptor\n");

    // Sphinx: #5
    /*******************************
     * Choose workspace limit based on available resources.
     *******************************/

    size_t freeMem, totalMem;
    HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
    uint64_t workspaceLimit = (uint64_t)((double)freeMem * 0.9);
    if (verbose) printf("Workspace limit = %lu\n", workspaceLimit);

    /*******************************
     * Find "optimal" contraction order and slicing
     *******************************/

    cutensornetContractionOptimizerConfig_t optimizerConfig;
    HANDLE_ERROR(cutensornetCreateContractionOptimizerConfig(handle, &optimizerConfig));

    // Set the desired number of hyper-samples (defaults to 0)
    int32_t num_hypersamples = 8;
    HANDLE_ERROR(cutensornetContractionOptimizerConfigSetAttribute(
      handle, optimizerConfig, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES,
      &num_hypersamples, sizeof(num_hypersamples)));

    // Create contraction optimizer info and find an optimized contraction path
    cutensornetContractionOptimizerInfo_t optimizerInfo;
    HANDLE_ERROR(cutensornetCreateContractionOptimizerInfo(handle, descNet, &optimizerInfo));

    HANDLE_ERROR(cutensornetContractionOptimize(handle, descNet, optimizerConfig, workspaceLimit,
                                                optimizerInfo));

    // Query the number of slices the tensor network execution will be split into
    int64_t numSlices = 0;
    HANDLE_ERROR(cutensornetContractionOptimizerInfoGetAttribute(
      handle, optimizerInfo, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES, &numSlices,
      sizeof(numSlices)));
    assert(numSlices > 0);

    if (verbose) printf("Found an optimized contraction path using cuTensorNet optimizer\n");

    // Sphinx: #6
    /*******************************
     * Create workspace descriptor, allocate workspace, and set it.
     *******************************/

    cutensornetWorkspaceDescriptor_t workDesc;
    HANDLE_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));

    int64_t requiredWorkspaceSize = 0;
    HANDLE_ERROR(
      cutensornetWorkspaceComputeContractionSizes(handle, descNet, optimizerInfo, workDesc));

    HANDLE_ERROR(cutensornetWorkspaceGetMemorySize(
      handle, workDesc, CUTENSORNET_WORKSIZE_PREF_MIN, CUTENSORNET_MEMSPACE_DEVICE,
      CUTENSORNET_WORKSPACE_SCRATCH, &requiredWorkspaceSize));

    void* work = nullptr;
    HANDLE_CUDA_ERROR(cudaMalloc(&work, requiredWorkspaceSize));

    HANDLE_ERROR(cutensornetWorkspaceSetMemory(handle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                                               CUTENSORNET_WORKSPACE_SCRATCH, work,
                                               requiredWorkspaceSize));

    if (verbose) printf("Allocated and set up the GPU workspace\n");

    // Sphinx: #7
    /*******************************
     * Initialize the pairwise contraction plan (for cuTENSOR).
     *******************************/

    cutensornetContractionPlan_t plan;
    HANDLE_ERROR(cutensornetCreateContractionPlan(handle, descNet, optimizerInfo, workDesc, &plan));

    /*******************************
     * Optional: Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel
     *           for each pairwise tensor contraction.
     *******************************/
    cutensornetContractionAutotunePreference_t autotunePref;
    HANDLE_ERROR(cutensornetCreateContractionAutotunePreference(handle, &autotunePref));

    const int numAutotuningIterations = 5;  // may be 0
    HANDLE_ERROR(cutensornetContractionAutotunePreferenceSetAttribute(
      handle, autotunePref, CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS,
      &numAutotuningIterations, sizeof(numAutotuningIterations)));

    // Modify the plan again to find the best pair-wise contractions
    HANDLE_ERROR(cutensornetContractionAutotune(handle, plan, rawDataIn_d, R_d, workDesc,
                                                autotunePref, stream));

    HANDLE_ERROR(cutensornetDestroyContractionAutotunePreference(autotunePref));

    if (verbose)
      printf("Created a contraction plan for cuTensorNet and optionally auto-tuned it\n");

    // Sphinx: #8
    /**********************
     * Execute the tensor network contraction
     **********************/

    // Create a cutensornetSliceGroup_t object from a range of slice IDs
    cutensornetSliceGroup_t sliceGroup{};
    HANDLE_ERROR(cutensornetCreateSliceGroupFromIDRange(handle, 0, numSlices, 1, &sliceGroup));

    GPUTimer timer{stream};
    double minTimeCUTENSORNET = 1e100;
    const int numRuns = 3;  // number of repeats to get stable performance results
    for (int i = 0; i < numRuns; ++i) {
      // HANDLE_CUDA_ERROR( cudaMemcpy(R_d, R, sizeR, cudaMemcpyHostToDevice) ); // restore the
      // output tensor on GPU HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );

      /*
       * Contract all slices of the tensor network
       */
      timer.start();

      int32_t accumulateOutput = 0;  // output tensor data will be overwritten
      HANDLE_ERROR(cutensornetContractSlices(
        handle, plan, rawDataIn_d, R_d, accumulateOutput, workDesc,
        sliceGroup,  // slternatively, NULL can also be used to contract over all slices instead of
                     // specifying a sliceGroup object
        stream));

      // Synchronize and measure best timing
      auto time = timer.seconds();
      minTimeCUTENSORNET = (time > minTimeCUTENSORNET) ? minTimeCUTENSORNET : time;
    }

    if (verbose)
      printf("Contracted the tensor network, each slice used the same contraction plan\n");

    // Print the 1-norm of the output tensor (verification)
    // HANDLE_CUDA_ERROR( cudaStreamSynchronize(stream) );
    // HANDLE_CUDA_ERROR( cudaMemcpy(R, R_d, sizeR, cudaMemcpyDeviceToHost) ); // restore the output
    // tensor on Host double norm1 = 0.0; for (int64_t i = 0; i < elementsR; ++i) {
    //    norm1 += std::abs(R[i]);
    // }
    // if(verbose)
    //    printf("Computed the 1-norm of the output tensor: %e\n", norm1);

    /*************************/

    // Query the total Flop count for the tensor network contraction
    double flops{0.0};
    HANDLE_ERROR(cutensornetContractionOptimizerInfoGetAttribute(
      handle, optimizerInfo, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT, &flops,
      sizeof(flops)));

    if (verbose) {
      printf("Number of tensor network slices = %ld\n", numSlices);
      printf("Tensor network contraction time (ms) = %.3f\n", minTimeCUTENSORNET * 1000.f);
    }

    // Free cuTensorNet resources
    HANDLE_ERROR(cutensornetDestroySliceGroup(sliceGroup));
    HANDLE_ERROR(cutensornetDestroyContractionPlan(plan));
    HANDLE_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
    HANDLE_ERROR(cutensornetDestroyContractionOptimizerInfo(optimizerInfo));
    HANDLE_ERROR(cutensornetDestroyContractionOptimizerConfig(optimizerConfig));
    HANDLE_ERROR(cutensornetDestroyNetworkDescriptor(descNet));
    HANDLE_ERROR(cutensornetDestroy(handle));

    // Free Host memory resources
    // if (R) free(R);
    // if (D) free(D);
    // if (C) free(C);
    // if (B) free(B);
    // if (A) free(A);

    // if (rawDataIn_d[0]) cudaFree(rawDataIn_d[0]);
    // if (rawDataIn_d[1]) cudaFree(rawDataIn_d[1]);
    // if (rawDataIn_d[2]) cudaFree(rawDataIn_d[2]);
    // if (rawDataIn_d[3]) cudaFree(rawDataIn_d[3]);

    // Free GPU memory resources

    if (work) cudaFree(work);
    if (R_d) cudaFree(R_d);
    for (size_t i = 0; i < numInputs; i++)
      if (rawDataIn_d[i]) cudaFree(rawDataIn_d[i]);

    if (verbose) printf("Freed resources and exited\n");
  }

  // Retrive the metas,datas of Unitensors and pass to cutensornet.
  void callcuTensornet(UniTensor& res, std::vector<UniTensor>& uts, bool& verbose) {
    static_assert(sizeof(size_t) == sizeof(int64_t),
                  "Please build this sample on a 64-bit architecture!");

    int32_t numInputs = uts.size();

    std::vector<int32_t*> modesIn;
    std::vector<int64_t*> extentsIn;
    std::vector<int32_t> numModesIn;
    std::vector<int64_t*> stridesIn;
    std::vector<void*> rawDataIn_d;
    std::vector<int32_t> modes_res;
    std::map<std::string, int32_t> lblmap;

    int64_t lbl_int;
    for (size_t i = 0; i < numInputs; i++) {
      std::vector<int32_t> tmp_mode;
      for (size_t j = 0; j < uts[i].labels().size(); j++) {
        lblmap.insert(std::pair<std::string, int32_t>(uts[i].labels()[j], lbl_int));
        tmp_mode.push_back(lblmap[uts[i].labels()[j]]);
      }
      modesIn.push_back(tmp_mode.data());
    }

    for (size_t i = 0; i < res.labels().size(); i++) {
      modes_res.push_back(lblmap[res.labels()[i]]);
    }

    for (size_t i = 0; i < numInputs; i++) {
      // modesIn.push_back(uts[i].labels().data());
      extentsIn.push_back((int64_t*)uts[i].shape().data());
      numModesIn.push_back(uts[i].labels().size());
      stridesIn.push_back(NULL);
      rawDataIn_d.push_back((void*)uts[i].get_block_()._impl->storage()._impl->Mem);
    }

    void* R_d = (void*)res.get_block_()._impl->storage()._impl->Mem;

    cuTensornet_(numInputs, R_d, res.shape().size(), (int64_t*)res.shape().data(), modes_res.data(),
                 rawDataIn_d.data(), modesIn.data(), numModesIn.data(), extentsIn.data(),
                 stridesIn.data(), verbose);
  }

#endif

}  // namespace cytnx
#endif