
#include <stdlib.h>
#include <stdio.h>

#include <unordered_map>
#include <vector>
#include <cassert>

#include <cuda_runtime.h>
#include <cutensornet.h>

#include <cytnx.hpp>
#include "cutensornet.hpp"

#ifdef UNI_CUQUANTUM
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

  cutensornet::cutensornet() {
    type_mapper = std::vector<cudaDataType_t>(11);
    type_mapper[Type.ComplexDouble] = CUDA_C_64F;
    type_mapper[Type.ComplexFloat] = CUDA_C_32F;
    type_mapper[Type.Double] = CUDA_R_64F;
    type_mapper[Type.Float] = CUDA_R_32F;
    verbose = true;
  };

  void cutensornet::parseLabels(std::vector<std::string> res_label,
                                std::vector<std::vector<std::string>> &labels) {
    int64_t lbl_int = 0;

    extentsIn = std::vector<int64_t *>(labels.size());
    stridesIn = std::vector<int64_t *>(labels.size());
    rawDataIn_d = std::vector<void *>(labels.size());
    extentR = std::vector<int64_t>(res_label.size());

    // reversed tranversal the labels and extents because cuTensor is column-major by default
    for (size_t i = 0; i < labels.size(); i++) {
      tmp_modes.push_back(std::vector<int32_t>(labels[i].size()));
      tmp_extents.push_back(std::vector<int64_t>(labels[i].size()));
      for (size_t j = 0; j < labels[i].size(); j++) {
        lblmap.insert(
          std::pair<std::string, int64_t>(labels[i][labels[i].size() - 1 - j], lbl_int));
        tmp_modes[i][j] = (lblmap[labels[i][labels[i].size() - 1 - j]]);
        lbl_int += 1;
      }
      modesIn.push_back(tmp_modes[i].data());
      numModesIn.push_back(labels[i].size());
    }
    for (size_t i = 0; i < res_label.size(); i++) {
      modesR.push_back(lblmap[res_label[res_label.size() - 1 - i]]);
    }
    numInputs = labels.size();
    nmodeR = res_label.size();
  }

  void cutensornet::updateOut(UniTensor &res) {
    // reversed tranversal the labels and extents because cuTensor is column-major by default
    for (size_t i = 0; i < nmodeR; i++) extentR[i] = res.shape()[nmodeR - 1 - i];
    R_d = (void *)res.get_block_()._impl->storage()._impl->Mem;
    typeData = type_mapper[res.dtype()];
    typeCompute = CUTENSORNET_COMPUTE_64F;
  }

  void cutensornet::updateTensor(int idx, UniTensor &ut) {
    // reversed tranversal the labels and extents because cuTensor is column-major by default
    for (size_t j = 0; j < numModesIn[idx]; j++)
      tmp_extents[idx][j] = ut.shape()[numModesIn[idx] - 1 - j];
    extentsIn[idx] = tmp_extents[idx].data();
    stridesIn[idx] = NULL;
    rawDataIn_d[idx] = (void *)ut.get_block_()._impl->storage()._impl->Mem;
  }

  // void cutensornet::updateDatas(UniTensor &res, std::vector<UniTensor> &uts){
  //   int64_t lbl_int = 0;
  //   tmp_extents.clear();
  //   extentsIn.clear();
  //   stridesIn.clear();
  //   rawDataIn_d.clear();
  //   extentR.clear();
  //   for (size_t i = 0; i < numInputs; i++) {
  //     tmp_extents.push_back(std::vector<int64_t>());
  //     for (size_t j = 0; j < numModesIn[i]; j++) tmp_extents[i].push_back(uts[i].shape()[j]);
  //     // reverse the labels and extents because cuTensor is column-major by default
  //     std::reverse(tmp_extents[i].begin(), tmp_extents[i].end());
  //     extentsIn.push_back(tmp_extents[i].data());
  //     stridesIn.push_back(NULL);
  //     rawDataIn_d.push_back((void *)uts[i].get_block_()._impl->storage()._impl->Mem);
  //   }
  //   for (size_t i = 0; i < nmodeR; i++) {
  //     extentR.push_back(res.shape()[i]);
  //   }
  //   // reverse the labels and extents because cuTensor is column-major by default
  //   std::reverse(extentR.begin(), extentR.end());
  //   R_d = (void *)res.get_block_()._impl->storage()._impl->Mem;

  //   typeData = type_mapper[res.dtype()];
  //   typeCompute = CUTENSORNET_COMPUTE_64F;
  // }

  void cutensornet::checkVersion() {
    cuTensornetVersion = cutensornetGetVersion();
    if (verbose) std::cout << "Cutensornet version: " << cuTensornetVersion << std::endl;
  }

  void cutensornet::setDevice() {
    HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices));
    HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceId));
    if (verbose) {
      std::cout << "Cutensornet number of available devices =  " << numDevices << std::endl;
      std::cout << "Cutensornet set device id =  " << deviceId << std::endl;
      printf("===== Cutensornet device info ======\n");
      printf("GPU-name:%s\n", prop.name);
      printf("GPU-clock:%d\n", prop.clockRate);
      printf("GPU-memoryClock:%d\n", prop.memoryClockRate);
      printf("GPU-nSM:%d\n", prop.multiProcessorCount);
      printf("GPU-major:%d\n", prop.major);
      printf("GPU-minor:%d\n", prop.minor);
      printf("========================\n");
    }
  }

  void cutensornet::createStream() { cudaStreamCreate(&stream); }

  void cutensornet::createHandle() { HANDLE_ERROR(cutensornetCreate(&handle)); }

  void cutensornet::createNetworkDescriptor() {
    if (verbose) {
      std::cout << "Cutensornet number of input tensors =  " << numInputs << std::endl;
      for (int i = 0; i < numInputs; i++) {
        std::cout << "-----------------tensor :" << i << "----------------------" << std::endl;
        std::cout << " num modes = " << numModesIn[i] << std::endl;
        std::cout << " label : ";
        for (int j = 0; j < numModesIn[i]; j++) {
          std::cout << (modesIn[i][j]) << " ";
        }
        std::cout << std::endl;
        std::cout << " shape : ";
        for (int j = 0; j < numModesIn[i]; j++) {
          std::cout << (extentsIn[i][j]) << " ";
        }
        std::cout << std::endl;
      }
      std::cout << "---------------------- output ----------------------" << std::endl;
      std::cout << " label : ";
      for (int j = 0; j < nmodeR; j++) {
        std::cout << (modesR[j]) << " ";
      }
      std::cout << std::endl;
      std::cout << " shape : ";
      for (int j = 0; j < nmodeR; j++) {
        std::cout << (extentR[j]) << " ";
      }
      std::cout << std::endl;
    }
    HANDLE_ERROR(cutensornetCreateNetworkDescriptor(
      (const cutensornetHandle_t)handle, numInputs, (const int32_t *)numModesIn.data(),
      (const int64_t **)extentsIn.data(), (const int64_t **)stridesIn.data(),
      (const int32_t **)modesIn.data(), NULL, nmodeR, (const int64_t *)extentR.data(),
      /*stridesOut = */ NULL, (const int32_t *)modesR.data(), typeData, typeCompute, &descNet));
    if (verbose)
      printf("Initialized the cuTensorNet library and created a tensor network descriptor\n");
  }

  void cutensornet::getWorkspacelimit() {
    HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
    workspaceLimit = (uint64_t)((double)freeMem * 0.9);
    if (verbose) printf("Workspace limit = %lu\n", workspaceLimit);
  }

  void cutensornet::findOptimalOrder() {
    HANDLE_ERROR(cutensornetCreateContractionOptimizerConfig(handle, &optimizerConfig));

    HANDLE_ERROR(cutensornetContractionOptimizerConfigSetAttribute(
      handle, optimizerConfig, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES,
      &num_hypersamples, sizeof(num_hypersamples)));

    HANDLE_ERROR(cutensornetCreateContractionOptimizerInfo(handle, descNet, &optimizerInfo));

    HANDLE_ERROR(cutensornetContractionOptimize(handle, descNet, optimizerConfig, workspaceLimit,
                                                optimizerInfo));
    HANDLE_ERROR(cutensornetContractionOptimizerInfoGetAttribute(
      handle, optimizerInfo, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES, &numSlices,
      sizeof(numSlices)));
    assert(numSlices > 0);
    if (verbose) printf("Found an optimized contraction path using cuTensorNet optimizer\n");
  }

  void cutensornet::createWorkspaceDescriptor() {
    HANDLE_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));
    HANDLE_ERROR(
      cutensornetWorkspaceComputeContractionSizes(handle, descNet, optimizerInfo, workDesc));
    HANDLE_ERROR(cutensornetWorkspaceGetMemorySize(
      handle, workDesc, CUTENSORNET_WORKSIZE_PREF_MIN, CUTENSORNET_MEMSPACE_DEVICE,
      CUTENSORNET_WORKSPACE_SCRATCH, &requiredWorkspaceSize));
    HANDLE_CUDA_ERROR(cudaMalloc(&work, requiredWorkspaceSize));
    HANDLE_ERROR(cutensornetWorkspaceSetMemory(handle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                                               CUTENSORNET_WORKSPACE_SCRATCH, work,
                                               requiredWorkspaceSize));
    if (verbose) printf("Allocated and set up the GPU workspace\n");
  }

  void cutensornet::initializePlan() {
    HANDLE_ERROR(cutensornetCreateContractionPlan(handle, descNet, optimizerInfo, workDesc, &plan));
  }

  void cutensornet::autotune() {
    HANDLE_ERROR(cutensornetCreateContractionAutotunePreference(handle, &autotunePref));
    HANDLE_ERROR(cutensornetContractionAutotunePreferenceSetAttribute(
      handle, autotunePref, CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS,
      &numAutotuningIterations, sizeof(numAutotuningIterations)));

    // Modify the plan again to find the best pair-wise contractions
    HANDLE_ERROR(cutensornetContractionAutotune(handle, plan, rawDataIn_d.data(), R_d, workDesc,
                                                autotunePref, stream));

    HANDLE_ERROR(cutensornetDestroyContractionAutotunePreference(autotunePref));
    if (verbose)
      printf("Created a contraction plan for cuTensorNet and optionally auto-tuned it\n");
  }

  void cutensornet::executeContraction() {
    HANDLE_ERROR(cutensornetCreateSliceGroupFromIDRange(handle, 0, numSlices, 1, &sliceGroup));
    GPUTimer timer{stream};
    double minTimeCUTENSORNET = 1e100;
    const int numRuns = 3;  // number of repeats to get stable performance results
    for (int i = 0; i < numRuns; ++i) {
      /*
       * Contract all slices of the tensor network
       */
      timer.start();

      int32_t accumulateOutput = 0;  // output tensor data will be overwritten
      HANDLE_ERROR(cutensornetContractSlices(
        handle, plan, rawDataIn_d.data(), R_d, accumulateOutput, workDesc,
        sliceGroup,  // slternatively, NULL can also be used to contract over all slices
                     // instead of specifying a sliceGroup object
        stream));
      // Synchronize and measure best timing
      auto time = timer.seconds();
      minTimeCUTENSORNET = (time > minTimeCUTENSORNET) ? minTimeCUTENSORNET : time;
    }
    if (verbose)
      printf("Contracted the tensor network, each slice used the same contraction plan\n");

    HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
    // HANDLE_CUDA_ERROR( cudaMemcpy(R, R_d, sizeR, cudaMemcpyDeviceToHost) ); // restore the
    // output tensor on Host

    if (verbose) {
      printf("Number of tensor network slices = %ld\n", numSlices);
      printf("Tensor network contraction time (ms) = %.3f\n", minTimeCUTENSORNET * 1000.f);
    }
  }

  void cutensornet::QueryFlopCount() {
    // Query the total Flop count for the tensor network contraction
    double flops{0.0};
    HANDLE_ERROR(cutensornetContractionOptimizerInfoGetAttribute(
      handle, optimizerInfo, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT, &flops,
      sizeof(flops)));
  }

  void cutensornet::free() {
    // Free cuTensorNet resources
    HANDLE_ERROR(cutensornetDestroySliceGroup(sliceGroup));
    HANDLE_ERROR(cutensornetDestroyContractionPlan(plan));
    HANDLE_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
    HANDLE_ERROR(cutensornetDestroyContractionOptimizerInfo(optimizerInfo));
    HANDLE_ERROR(cutensornetDestroyContractionOptimizerConfig(optimizerConfig));
    HANDLE_ERROR(cutensornetDestroyNetworkDescriptor(descNet));
    HANDLE_ERROR(cutensornetDestroy(handle));
  }

  #endif

}  // namespace cytnx
#endif