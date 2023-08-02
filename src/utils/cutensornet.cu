
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

  cutensornet::cutensornet(){
    // this->extentR =  std::vector<int64_t>();
    // this->modesR = std::vector<int32_t>();
    // this->modesIn = std::vector<int32_t*>();
    // this->numModesIn = std::vector<int32_t>();
    // this->extentsIn = std::vector<int64_t*>();
    // this->stridesIn = std::vector<int64_t*>();
  };

  void cutensornet::initialize(UniTensor& res, std::vector<UniTensor>& uts, std::vector<std::vector<std::string>>& labels, bool verbose){
    static_assert(sizeof(size_t) == sizeof(int64_t),
                "Please build this sample on a 64-bit architecture!");
    int32_t numInputs = uts.size();
    // std::vector<int32_t*> modesIn(numInputs);
    // std::vector<int64_t*> extentsIn;
    // std::vector<int32_t> numModesIn;
    // std::vector<int64_t*> stridesIn;
    // std::vector<void*> rawDataIn_d;
    // std::vector<int32_t> modes_res;
    int64_t lbl_int = 0;
    for (size_t i = 0; i < numInputs; i++) {
      tmp_modes.push_back(std::vector<int32_t>());
      tmp_extents.push_back(std::vector<int64_t>());
      for (size_t j = 0; j < labels[i].size(); j++) {
        lblmap.insert(std::pair<std::string, int64_t>(labels[i][j], lbl_int));
        tmp_modes[i].push_back(lblmap[labels[i][j]]);
        tmp_extents[i].push_back(uts[i].shape()[j]);
        lbl_int+=1;
      }
    this->modesIn.push_back(tmp_modes[i].data());
    this->extentsIn.push_back(tmp_extents[i].data());
    this->numModesIn.push_back(labels[i].size());
    this->stridesIn.push_back(NULL);
    this->rawDataIn_d.push_back((void*)uts[i].get_block_()._impl->storage()._impl->Mem);
    }

    this->nmodeR = res.labels().size();
    for (size_t i = 0; i < this->nmodeR; i++) {
      this->modesR.push_back(lblmap[res.labels()[i]]);
      this->extentR.push_back(res.shape()[i]);
    }

    this->R_d = (void*)res.get_block_()._impl->storage()._impl->Mem;
    this->numInputs = numInputs;
    this->verbose = verbose;
  }

  // cutensornet::cutensornet(int32_t numInputs, void* R_d, int32_t nmodeR, int64_t* extentR,
  //                   int32_t* modesR, void* rawDataIn_d[], int32_t* modesIn[],
  //                   int32_t numModesIn[], int64_t* extentsIn[], int64_t* stridesIn[],
  //                   bool verbose){

  //     static_assert(sizeof(size_t) == sizeof(int64_t),
  //                 "Please build this sample on a 64-bit architecture!");

  //     this->numInputs = numInputs;
  //     this->R_d = R_d;
  //     this->nmodeR = nmodeR;
  //     this->extentR = extentR;
  //     this->modesR = modesR;
  //     this->rawDataIn_d = rawDataIn_d;
  //     this->modesIn = modesIn;
  //     this->numModesIn = numModesIn;
  //     this->extentsIn = extentsIn;
  //     this->stridesIn = stridesIn;
  //     this->verbose = verbose;

  // }

  
  void cutensornet::checkVersion(){
    this->cuTensornetVersion = cutensornetGetVersion();
    if(this->verbose) std::cout<<"Cutensornet version: "<< this->cuTensornetVersion<<std::endl;
  }

  void cutensornet::setDevice(){
    HANDLE_CUDA_ERROR(cudaGetDeviceCount(&this->numDevices));
    HANDLE_CUDA_ERROR(cudaSetDevice(this->deviceId));
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&this->prop, this->deviceId));
    if(this->verbose){
      std::cout<<"Cutensornet number of available devices =  "<< this->numDevices<<std::endl;
      std::cout<<"Cutensornet set device id =  "<< this->deviceId<<std::endl;
    } 
  }

  void cutensornet::setType(){
    typedef float floatType;
    this->typeData = CUDA_R_32F;
    this->typeCompute = CUTENSORNET_COMPUTE_32F;
    if(this->verbose) {
      std::cout<<"Cutensornet set type data =  "<< this->typeData<<std::endl;
      std::cout<<"Cutensornet set type compute =  "<< this->typeCompute<<std::endl;
    }
  }

  void cutensornet::createStream(){
    cudaStreamCreate(&this->stream);
  }

  void cutensornet::createHandle(){
    HANDLE_ERROR(cutensornetCreate(&this->handle));
  }

  void cutensornet::createNetworkDescriptor(){
    if(this->verbose) {
      std::cout<<"Cutensornet number of input tensors =  "<<this->numInputs<<std::endl;
      for(int i = 0; i < this->numInputs; i++){
        std::cout<<"-----------------tensor :"<<i<<"----------------------"<<std::endl;
        std::cout<<" num modes = " << this->numModesIn[i]<<std::endl;
        for(int j = 0; j< this->numModesIn[i]; j++){
          std::cout<<(this->modesIn[i][j])<<" ";
        }
        std::cout<<std::endl; 
        for(int j = 0; j< this->numModesIn[i]; j++){
          std::cout<<(this->extentsIn[i][j])<<" ";
        }
        std::cout<<std::endl; 
      }
      std::cout<<"res ================="<<std::endl;
      for(int j = 0; j< this->nmodeR; j++){
          std::cout<<(this->modesR[j])<<" ";
      }
      for(int j = 0; j< this->nmodeR; j++){
          std::cout<<(this->extentR[j])<<" ";
      }
    }
    HANDLE_ERROR(cutensornetCreateNetworkDescriptor(
    (const cutensornetHandle_t) this->handle, this->numInputs, (const int32_t *)this->numModesIn.data(), (const int64_t **)this->extentsIn.data(), (const int64_t **)this->stridesIn.data(), (const int32_t **)this->modesIn.data(), NULL, this->nmodeR, (const int64_t *)this->extentR.data(),
    /*stridesOut = */ NULL, (const int32_t *)this->modesR.data(), this->typeData, this->typeCompute, &this->descNet));
  }

  void cutensornet::getWorkspacelimit(){
    HANDLE_CUDA_ERROR(cudaMemGetInfo(&this->freeMem, &this->totalMem));
    this->workspaceLimit = (uint64_t)((double)this->freeMem * 0.9);
  }

  void cutensornet::findOptimalOrder(){
    HANDLE_ERROR(cutensornetCreateContractionOptimizerConfig(this->handle, &this->optimizerConfig));

    HANDLE_ERROR(cutensornetContractionOptimizerConfigSetAttribute(
      this->handle, this->optimizerConfig, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES,
      &this->num_hypersamples, sizeof(this->num_hypersamples)));

    HANDLE_ERROR(cutensornetCreateContractionOptimizerInfo(this->handle, this->descNet, &this->optimizerInfo));

    HANDLE_ERROR(cutensornetContractionOptimize(this->handle, this->descNet, this->optimizerConfig, this->workspaceLimit,
                                                this->optimizerInfo));
  }
    
  void cutensornet::querySlices(){
    HANDLE_ERROR(cutensornetContractionOptimizerInfoGetAttribute(
      this->handle, this->optimizerInfo, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES, &this->numSlices,
      sizeof(this->numSlices)));
    assert(this->numSlices > 0);
  }

  void cutensornet::createWorkspaceDescriptor(){
    HANDLE_ERROR(cutensornetCreateWorkspaceDescriptor(this->handle, &this->workDesc));
    HANDLE_ERROR(
      cutensornetWorkspaceComputeContractionSizes(this->handle, this->descNet, this->optimizerInfo, this->workDesc));
    HANDLE_ERROR(cutensornetWorkspaceGetMemorySize(
      this->handle, this->workDesc, CUTENSORNET_WORKSIZE_PREF_MIN, CUTENSORNET_MEMSPACE_DEVICE,
      CUTENSORNET_WORKSPACE_SCRATCH, &this->requiredWorkspaceSize));
    HANDLE_CUDA_ERROR(cudaMalloc(&this->work, this->requiredWorkspaceSize));
    HANDLE_ERROR(cutensornetWorkspaceSetMemory(this->handle, this->workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                                               CUTENSORNET_WORKSPACE_SCRATCH, this->work,
                                               this->requiredWorkspaceSize));
  }

  void cutensornet::initializePlan(){
    HANDLE_ERROR(cutensornetCreateContractionPlan(this->handle, this->descNet, this->optimizerInfo, this->workDesc, &this->plan));
  }

  void cutensornet::autotune(){
    HANDLE_ERROR(cutensornetCreateContractionAutotunePreference(this->handle, &this->autotunePref));
    HANDLE_ERROR(cutensornetContractionAutotunePreferenceSetAttribute(
      this->handle, this->autotunePref, CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS,
      &this->numAutotuningIterations, sizeof(this->numAutotuningIterations)));

    // Modify the plan again to find the best pair-wise contractions
    HANDLE_ERROR(cutensornetContractionAutotune(this->handle, this->plan, this->rawDataIn_d.data(), this->R_d, this->workDesc,
                                                this->autotunePref, this->stream));

    HANDLE_ERROR(cutensornetDestroyContractionAutotunePreference(this->autotunePref));

  }

  void cutensornet::executeContraction(){
    HANDLE_ERROR(cutensornetCreateSliceGroupFromIDRange(this->handle, 0, this->numSlices, 1, &this->sliceGroup));
       GPUTimer timer{this->stream};
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
        this->handle, this->plan, this->rawDataIn_d.data(), this->R_d, accumulateOutput, this->workDesc,
        this->sliceGroup,  // slternatively, NULL can also be used to contract over all slices instead of
                     // specifying a sliceGroup object
        this->stream));
      // Synchronize and measure best timing
      auto time = timer.seconds();
      minTimeCUTENSORNET = (time > minTimeCUTENSORNET) ? minTimeCUTENSORNET : time;
    }
  }

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
  void callcuTensornet(UniTensor& res, std::vector<UniTensor>& uts, bool verbose) {
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