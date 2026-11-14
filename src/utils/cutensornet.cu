#include <stdio.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cassert>
#include <cytnx.hpp>
#include "cutensornet.hpp"

#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM
    #include <cutensornet.h>
    #include <cuda_runtime.h>
  #endif
#endif

namespace cytnx {
#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM

    // A failed cuTensorNet or CUDA call raises cytnx::error; the cutensornet object that made the
    // call releases what it holds while the exception unwinds.
    #define HANDLE_ERROR(x)                                                                     \
      {                                                                                         \
        const auto err = x;                                                                     \
        cytnx_error_msg(err != CUTENSORNET_STATUS_SUCCESS, "[ERROR][cutensornet] %s: %s\n", #x, \
                        cutensornetGetErrorString(err));                                        \
      };

    #define HANDLE_CUDA_ERROR(x)                                                               \
      {                                                                                        \
        const auto err = x;                                                                    \
        cytnx_error_msg(err != cudaSuccess, "[ERROR][cutensornet] CUDA error in %s: %s\n", #x, \
                        cudaGetErrorString(err));                                              \
      };

    // For the destructor, which must not throw: report and keep releasing the rest.
    #define WARN_ON_ERROR(x)                                                                    \
      {                                                                                         \
        const auto err = x;                                                                     \
        cytnx_warning_msg(err != CUTENSORNET_STATUS_SUCCESS, "[WARNING][cutensornet] %s: %s\n", \
                          #x, cutensornetGetErrorString(err));                                  \
      };

    #define WARN_ON_CUDA_ERROR(x)                                                                  \
      {                                                                                            \
        const auto err = x;                                                                        \
        cytnx_warning_msg(err != cudaSuccess, "[WARNING][cutensornet] CUDA error in %s: %s\n", #x, \
                          cudaGetErrorString(err));                                                \
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

  cutensornet::cutensornet() { verbose = false; };

  // Everything is released in reverse order of creation; the handle goes after all objects that
  // were created through it.
  cutensornet::~cutensornet() {
    if (sliceGroup != nullptr) WARN_ON_ERROR(cutensornetDestroySliceGroup(sliceGroup));
    if (autotunePref != nullptr)
      WARN_ON_ERROR(cutensornetDestroyContractionAutotunePreference(autotunePref));
    if (plan != nullptr) WARN_ON_ERROR(cutensornetDestroyContractionPlan(plan));
    if (workDesc != nullptr) WARN_ON_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
    if (work != nullptr) WARN_ON_CUDA_ERROR(cudaFree(work));
    if (optimizerConfig != nullptr)
      WARN_ON_ERROR(cutensornetDestroyContractionOptimizerConfig(optimizerConfig));
    if (ownsOptimizerInfo && optimizerInfo != nullptr)
      WARN_ON_ERROR(cutensornetDestroyContractionOptimizerInfo(optimizerInfo));
    if (descNet != nullptr) WARN_ON_ERROR(cutensornetDestroyNetworkDescriptor(descNet));
    if (handle != nullptr) WARN_ON_ERROR(cutensornetDestroy(handle));
    if (stream != nullptr) WARN_ON_CUDA_ERROR(cudaStreamDestroy(stream));
  }

  void cutensornet::parseLabels(std::vector<int64_t> &res_label,
                                std::vector<std::vector<int64_t>> &labels) {
    // int64_t lbl_int = 0;

    extentsIn = std::vector<int64_t *>(labels.size());
    stridesIn = std::vector<int64_t *>(labels.size());
    tns = std::vector<UniTensor>(labels.size());
    extentR = std::vector<int64_t>(res_label.size());
    tmp_modes = std::vector<std::vector<int32_t>>(labels.size());
    tmp_extents = std::vector<std::vector<int64_t>>(labels.size());
    lblmap = std::map<std::string, int32_t>();
    modesIn = std::vector<int32_t *>(labels.size());
    numModesIn = std::vector<int32_t>(labels.size());
    modesR = std::vector<int32_t>(labels.size());
    rawDataIn_d = std::vector<void *>(labels.size());

    for (std::size_t i = 0; i < labels.size(); i++) {
      tmp_modes[i] = std::vector<int32_t>(labels[i].size());
      tmp_extents[i] = std::vector<int64_t>(labels[i].size());
      for (std::size_t j = 0; j < labels[i].size(); j++) {
        tmp_modes[i][j] = (labels[i][labels[i].size() - 1 - j]);
      }
      modesIn[i] = tmp_modes[i].data();
      numModesIn[i] = labels[i].size();
    }
    for (std::size_t i = 0; i < res_label.size(); i++) {
      modesR[i] = res_label[res_label.size() - 1 - i];
    }
    numInputs = labels.size();
    nmodeR = res_label.size();
  }

  void cutensornet::set_output_extents(std::vector<cytnx_uint64> &outshape) {
    extentR = std::vector<int64_t>(outshape.size());
    // reversed tranversal the labels and extents because cuTensor is column-major by default
    for (std::size_t i = 0; i < outshape.size(); i++)
      extentR[i] = outshape[outshape.size() - 1 - i];
  }

  void cutensornet::setOutputMem(UniTensor &res) {
    R_d = (void *)res.get_block_()._impl->storage()._impl->data();
  }

  void cutensornet::setInputMem(std::vector<UniTensor> &uts) {
    // reversed tranversal the labels and extents because cuTensor is column-major by default
    rawDataIn_d = std::vector<void *>(uts.size());
    tns = std::vector<UniTensor>(uts.size());
    for (int idx = 0; idx < uts.size(); idx++) {
      if (uts[idx].is_contiguous())
        rawDataIn_d[idx] = (void *)uts[idx].get_block_()._impl->storage()._impl->data();
      else
        rawDataIn_d[idx] =
          (void *)uts[idx].get_block_().contiguous()._impl->storage()._impl->data();
      if (uts[idx].is_contiguous())
        tns[idx] = uts[idx];
      else
        tns[idx] = uts[idx].contiguous();
      rawDataIn_d[idx] = tns[idx].get_block_()._impl->storage()._impl->data();
    }
  }

  void cutensornet::set_extents(std::vector<UniTensor> &uts) {
    // reversed tranversal the labels and extents because cuTensor is column-major by default
    for (std::size_t idx = 0; idx < numInputs; idx++) {
      for (std::size_t j = 0; j < numModesIn[idx]; j++) {
        tmp_extents[idx][j] = uts[idx].shape()[numModesIn[idx] - 1 - j];
      }
      extentsIn[idx] = tmp_extents[idx].data();
      stridesIn[idx] = nullptr;
    }
  }

  void cutensornet::setContractionPath(std::vector<std::pair<int64_t, int64_t>> einsum_path) {
    // cutensornetContractionOptimizerConfigAttributes_t attr
    // Not a std::vector: cutensornetNodePair_t is an unnamed struct, and nvcc 12.9 aborts with an
    // internal error when it instantiates the element destructor call of the vector.
    auto pairs = std::make_unique<cutensornetNodePair_t[]>(einsum_path.size());
    for (std::size_t i = 0; i < einsum_path.size(); i++) {
      pairs[i].first = einsum_path[i].first;
      pairs[i].second = einsum_path[i].second;
    }
    cutensornetContractionPath_t path;
    path.numContractions = einsum_path.size();
    path.data = pairs.get();
    HANDLE_ERROR(cutensornetContractionOptimizerInfoSetAttribute(
      handle, optimizerInfo, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH, &path, sizeof(path)));
  }

  std::vector<std::pair<int64_t, int64_t>> cutensornet::getContractionPath() {
    auto pairs = std::make_unique<cutensornetNodePair_t[]>(numInputs);
    cutensornetContractionPath_t path;
    path.numContractions = 0;
    path.data = pairs.get();

    HANDLE_ERROR(cutensornetContractionOptimizerInfoGetAttribute(
      handle, optimizerInfo, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH, &path, sizeof(path)));

    std::vector<std::pair<int64_t, int64_t>> einsum_path;

    for (int i = 0; i < path.numContractions; i++) {
      einsum_path.push_back(
        std::pair<int64_t, int64_t>((int64_t)path.data[i].first, (int64_t)path.data[i].second));
    }
    HANDLE_ERROR(cutensornetContractionOptimizerInfoGetAttribute(
      handle, optimizerInfo, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES, &numSlices,
      sizeof(numSlices)));  // get numslices
    assert(numSlices > 0);

    return einsum_path;
  }

  void cutensornet::checkVersion() {
    cuTensornetVersion = cutensornetGetVersion();
    if (verbose) std::cout << "Cutensornet version: " << cuTensornetVersion << std::endl;
  }

  void cutensornet::setDevice(int id) {
    HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices));
    HANDLE_CUDA_ERROR(cudaSetDevice(id));
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, id));
    if (verbose) {
      std::cout << "Cutensornet number of available devices =  " << numDevices << std::endl;
      std::cout << "Cutensornet set device id =  " << deviceId << std::endl;
      printf("===== Cutensornet device info ======\n");
      printf("GPU-name:%s\n", prop.name);
    #if (CUDART_VERSION < 13000)
      printf("GPU-clock:%d\n", prop.clockRate);
      printf("GPU-memoryClock:%d\n", prop.memoryClockRate);
    #endif
      printf("GPU-nSM:%d\n", prop.multiProcessorCount);
      printf("GPU-major:%d\n", prop.major);
      printf("GPU-minor:%d\n", prop.minor);
      printf("========================\n");
    }
  }

  void cutensornet::createStream() { HANDLE_CUDA_ERROR(cudaStreamCreate(&stream)); }

  void cutensornet::createHandle() { HANDLE_ERROR(cutensornetCreate(&handle)); }

  // cutensornetNetworkDescriptor_t cutensornet::createNetworkDescriptor(std::vector<std::string>
  // &res_label,
  //                               std::vector<std::vector<std::string>> &labels,
  //                               std::vector<UniTensor> &uts) {
  void cutensornet::createNetworkDescriptor(unsigned int dtype) {
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
    if (dtype == Type.ComplexDouble) {
      typeData = CUDA_C_64F;
      typeCompute = CUTENSORNET_COMPUTE_64F;
    } else if (dtype == Type.ComplexFloat) {
      typeData = CUDA_C_32F;
      typeCompute = CUTENSORNET_COMPUTE_32F;
    } else if (dtype == Type.Double) {
      typeData = CUDA_R_64F;
      typeCompute = CUTENSORNET_COMPUTE_64F;
    } else if (dtype == Type.Float) {
      typeData = CUDA_R_32F;
      typeCompute = CUTENSORNET_COMPUTE_32F;
    } else {
      cytnx_error_msg(true, "[ERROR][cutensornet] dtype %s cannot be contracted by cuTensorNet.\n",
                      Type.getname(dtype).c_str());
    }
    HANDLE_ERROR(cutensornetCreateNetworkDescriptor(
      (const cutensornetHandle_t)handle, numInputs, (const int32_t *)numModesIn.data(),
      (const int64_t **)extentsIn.data(), (const int64_t **)stridesIn.data(),
      (const int32_t **)modesIn.data(), nullptr, nmodeR, (const int64_t *)extentR.data(),
      /*stridesOut = */ nullptr, (const int32_t *)modesR.data(), typeData, typeCompute, &descNet));
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
      &num_hypersamples,
      sizeof(num_hypersamples)));  // OptimizerConfig means the pathfinder's config.

    createOptimizerInfo();

    HANDLE_ERROR(cutensornetContractionOptimize(handle, descNet, optimizerConfig, workspaceLimit,
                                                optimizerInfo));

    if (verbose) printf("Found an optimized contraction path using cuTensorNet optimizer\n");

    HANDLE_ERROR(cutensornetDestroyContractionOptimizerConfig(optimizerConfig));
    optimizerConfig = nullptr;
  }

  void cutensornet::createOptimizerInfo() {
    HANDLE_ERROR(cutensornetCreateContractionOptimizerInfo(
      handle, descNet,
      &optimizerInfo));  // OptimizerInfo means the info of optimized path and slices.
    ownsOptimizerInfo = true;
  }

  void cutensornet::useOptimizerInfoOf(const cutensornet &owner) {
    optimizerInfo = owner.optimizerInfo;
    ownsOptimizerInfo = false;
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
    autotunePref = nullptr;
    if (verbose)
      printf("Created a contraction plan for cuTensorNet and optionally auto-tuned it\n");
  }

  void cutensornet::executeContraction() {
    HANDLE_ERROR(cutensornetContractionOptimizerInfoGetAttribute(
      handle, optimizerInfo, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES, &numSlices,
      sizeof(numSlices)));
    HANDLE_ERROR(cutensornetCreateSliceGroupFromIDRange(handle, 0, numSlices, 1, &sliceGroup));

    HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
    GPUTimer timer{stream};
    double minTimeCUTENSORNET = 1e100;
    const int numRuns = 10;  // number of repeats to get stable performance results

    // for (int i = 0; i < numRuns; ++i) {
    /*
     * Contract all slices of the tensor network
     */
    timer.start();

    int32_t accumulateOutput = 0;  // output tensor data will be overwritten
    // HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
    HANDLE_ERROR(cutensornetContractSlices(
      handle, plan, rawDataIn_d.data(), R_d, accumulateOutput, workDesc,
      sliceGroup,  // alternatively, nullptr can also be used to contract over all
                   // slices instead of specifying a sliceGroup object
      stream));
    HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
    // Synchronize and measure best timing
    auto time = timer.seconds();
    minTimeCUTENSORNET = (time > minTimeCUTENSORNET) ? minTimeCUTENSORNET : time;

    if (verbose)
      printf("Contracted the tensor network, each slice used the same contraction plan\n");

    // HANDLE_CUDA_ERROR( cudaMemcpy(R, R_d, sizeR, cudaMemcpyDeviceToHost) ); // restore the
    // output tensor on Host

    if (verbose) {
      printf("Number of tensor network slices = %ld\n", numSlices);
      printf("Tensor network contraction time (ms) = %.3f\n", minTimeCUTENSORNET * 1000.f);
    }
    HANDLE_ERROR(cutensornetDestroySliceGroup(sliceGroup));
    sliceGroup = nullptr;
  }

  void cutensornet::QueryFlopCount() {
    // Query the total Flop count for the tensor network contraction
    double flops{0.0};
    HANDLE_ERROR(cutensornetContractionOptimizerInfoGetAttribute(
      handle, optimizerInfo, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT, &flops,
      sizeof(flops)));
  }
  #endif
#endif
}  // namespace cytnx
