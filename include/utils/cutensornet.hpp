#ifndef __cutensornet_H_
#define __cutensornet_H_

#ifdef UNI_CUQUANTUM

  #include "Type.hpp"
  #include "cytnx_error.hpp"
  #include <cuda_runtime.h>
  #include <cutensornet.h>

namespace cytnx {

  class cutensornet {
   private:
    std::vector<cudaDataType_t> type_mapper;

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
    int32_t num_hypersamples;

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
    bool verbose;
    std::vector<void *> rawDataIn_d;
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
    cutensornet();
    void parseLabels(std::vector<std::string> res_label,
                     std::vector<std::vector<std::string>> &labels);
    void updateOut(UniTensor &res);
    void updateTensor(int idx, UniTensor &ut);
    // void updateDatas(UniTensor &res, std::vector<UniTensor> &uts);
    void checkVersion();
    void setDevice();
    void createStream();
    void createHandle();
    void createNetworkDescriptor();
    void getWorkspacelimit();
    void findOptimalOrder();
    void querySlices();
    void createWorkspaceDescriptor();
    void initializePlan();
    void autotune();
    void executeContraction();
    void QueryFlopCount();
    void free();
  };

}  // namespace cytnx

#endif
#endif
