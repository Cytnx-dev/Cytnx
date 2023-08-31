#ifndef __cutensornet_H_
#define __cutensornet_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
// #include "Tensor.hpp"
// #include "UniTensor.hpp"

#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM
    #include <cutensornet.h>
    #include <cuda_runtime.h>
  #endif
#endif

namespace cytnx {

#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM

  class cutensornet {
   private:
    // std::vector<cudaDataType_t> type_mapper;

    // cuTensorNet version
    size_t cuTensornetVersion;

    // GPU device
    int numDevices{0};
    int deviceId = 0;
    // Device Properties
    cudaDeviceProp prop;

    // data types
    cudaDataType_t typeData;  // CUDA_R_32F;
    cutensornetComputeType_t typeCompute;  // CUTENSORNET_COMPUTE_32F;

    // stream
    cudaStream_t stream;

    // cutensornet handle
    cutensornetHandle_t handle;

    // network descriptor
    cutensornetNetworkDescriptor_t descNet;

    // mem info
    size_t freeMem, totalMem;
    uint64_t workspaceLimit;

    // optimizer config
    cutensornetContractionOptimizerConfig_t optimizerConfig;

    // number of hypersamples
    int32_t num_hypersamples = 8;

    // optimizer info
    cutensornetContractionOptimizerInfo_t optimizerInfo;

    // the number of slices the tensor network execution will be split into
    int64_t numSlices;

    // workspace descriptor
    cutensornetWorkspaceDescriptor_t workDesc;

    // required workspace
    int64_t requiredWorkspaceSize = 0;
    void *work = nullptr;

    // contraction plan
    cutensornetContractionPlan_t plan;

    // auto tune
    cutensornetContractionAutotunePreference_t autotunePref;
    int numAutotuningIterations = 5;  // may be 0

    // Create a cutensornetSliceGroup_t object from a range of slice IDs
    cutensornetSliceGroup_t sliceGroup{};

    // input datas
    int32_t numInputs;
    void *R_d;
    int32_t nmodeR;
    bool verbose = false;  // For DEBUG use

    std::vector<void *> rawDataIn_d;
    std::vector<UniTensor> tns;
    std::vector<int64_t> extentR;
    std::vector<int32_t> modesR;
    std::vector<int32_t *> modesIn;
    std::vector<int32_t> numModesIn;
    std::vector<int64_t *> extentsIn;
    std::vector<int64_t *> stridesIn;

    std::map<std::string, int32_t> lblmap;
    std::vector<std::vector<int32_t>> tmp_modes;
    std::vector<std::vector<int64_t>> tmp_extents;

   public:
    UniTensor out;
    cutensornet();
    // ~cutensornet();
    void parseLabels(std::vector<int64_t> &res_label, std::vector<std::vector<int64_t>> &labels);
    void setOutputMem(UniTensor &res);
    void setInputMem(std::vector<UniTensor> &uts);
    void set_extents(std::vector<UniTensor> &uts);
    void set_output_extents(std::vector<cytnx_uint64> &outshape);
    void checkVersion();
    void setDevice(int id);
    void createStream();
    void createHandle();
    cutensornetWorkspaceDescriptor_t createNetworkDescriptor();
    void getWorkspacelimit();
    cutensornetContractionOptimizerInfo_t findOptimalOrder();
    cutensornetContractionOptimizerInfo_t createOptimizerInfo();
    void createWorkspaceDescriptor();
    void initializePlan();
    void autotune();
    void executeContraction();

    void setContractionPath(std::vector<std::pair<int64_t, int64_t>> einsum_path);
    std::vector<std::pair<int64_t, int64_t>> getContractionPath();

    void setNetworkDescriptor(cutensornetNetworkDescriptor_t in);
    void setOptimizerInfo(cutensornetContractionOptimizerInfo_t in);
    // void createContractor();

    void QueryFlopCount();

    void freePlan();
    void freeOptimizer();
    void freeWorkspaceDescriptor();
    void freeNetworkDescriptor();
    void freeHandle();
  };

  #endif
#endif
}  // namespace cytnx

#endif
