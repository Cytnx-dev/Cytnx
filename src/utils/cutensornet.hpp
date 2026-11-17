#ifndef CYTNX_UTILS_CUTENSORNET_H_
#define CYTNX_UTILS_CUTENSORNET_H_

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Type.hpp"
#include "UniTensor.hpp"
#include "cytnx_error.hpp"

#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM
    #include <cutensornet.h>
    #include <cuda_runtime.h>
  #endif
#endif

namespace cytnx {

#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM

  // One cuTensorNet session: a library handle and everything created through it. The object owns
  // what it creates and releases it in its destructor, so a session lasts exactly as long as the
  // object and cannot be copied.
  class cutensornet {
   private:
    // std::vector<cudaDataType_t> type_mapper;

    // cuTensorNet version
    std::size_t cuTensornetVersion;

    // GPU device
    int numDevices{0};
    int deviceId = 0;
    // Device Properties
    cudaDeviceProp prop;

    // data types
    cudaDataType_t typeData;  // CUDA_R_32F;
    cutensornetComputeType_t typeCompute;  // CUTENSORNET_COMPUTE_32F;

    // stream
    cudaStream_t stream = nullptr;

    // cutensornet handle
    cutensornetHandle_t handle = nullptr;

    // network descriptor
    cutensornetNetworkDescriptor_t descNet = nullptr;

    // mem info
    std::size_t freeMem, totalMem;
    uint64_t workspaceLimit;

    // optimizer config
    cutensornetContractionOptimizerConfig_t optimizerConfig = nullptr;

    // number of hypersamples
    int32_t num_hypersamples = 8;

    // optimizer info; owned unless it was borrowed with useOptimizerInfoOf()
    cutensornetContractionOptimizerInfo_t optimizerInfo = nullptr;
    bool ownsOptimizerInfo = false;

    // the number of slices the tensor network execution will be split into
    int64_t numSlices;

    // workspace descriptor
    cutensornetWorkspaceDescriptor_t workDesc = nullptr;

    // required workspace
    int64_t requiredWorkspaceSize = 0;
    void *work = nullptr;

    // contraction plan
    cutensornetContractionPlan_t plan = nullptr;

    // auto tune
    cutensornetContractionAutotunePreference_t autotunePref = nullptr;
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
    ~cutensornet();
    cutensornet(const cutensornet &) = delete;
    cutensornet &operator=(const cutensornet &) = delete;
    void parseLabels(std::vector<int64_t> &res_label, std::vector<std::vector<int64_t>> &labels);
    void setOutputMem(UniTensor &res);
    void setInputMem(std::vector<UniTensor> &uts);
    void set_extents(std::vector<UniTensor> &uts);
    void set_output_extents(std::vector<cytnx_uint64> &outshape);
    void checkVersion();
    void setDevice(int id);
    void createStream();
    void createHandle();
    // dtype is the common dtype of every input tensor and of the output; it must be one of
    // Type.ComplexDouble, Type.ComplexFloat, Type.Double, Type.Float.
    void createNetworkDescriptor(unsigned int dtype);
    void getWorkspacelimit();
    void findOptimalOrder();
    void createOptimizerInfo();
    // Contract with the optimizer info that `owner` holds; `owner` has to outlive this object.
    void useOptimizerInfoOf(const cutensornet &owner);
    void createWorkspaceDescriptor();
    void initializePlan();
    void autotune();
    void executeContraction();

    void setContractionPath(std::vector<std::pair<int64_t, int64_t>> einsum_path);
    std::vector<std::pair<int64_t, int64_t>> getContractionPath();

    void QueryFlopCount();
  };

  #endif
#endif
}  // namespace cytnx

#endif  // CYTNX_UTILS_CUTENSORNET_H_
