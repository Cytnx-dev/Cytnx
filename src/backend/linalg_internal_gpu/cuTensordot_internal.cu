#ifdef UNI_CUTENSOR
  #include "cuTensordot_internal.hpp"
  #include "cytnx_error.hpp"
  #include "Type.hpp"
  #include "backend/lapack_wrapper.hpp"
  #include <cutensor.h>

  #define HANDLE_ERROR(x)                                                        \
    {                                                                            \
      const cutensorStatus_t err = x;                                            \
      if (err != CUTENSOR_STATUS_SUCCESS) {                                      \
        printf("Error in line %d: %s\n", __LINE__, cutensorGetErrorString(err)); \
        exit(-1);                                                                \
      }                                                                          \
    };

  #define HANDLE_CUDA_ERROR(x)                           \
    {                                                    \
      const auto err = x;                                \
      if (err != cudaSuccess) {                          \
        printf("Error in line %d: %s\n", __LINE__, err); \
        exit(-1);                                        \
      }                                                  \
    };

namespace cytnx {

  namespace linalg_internal {

    inline void _cuTensordot_internal(Tensor &out, const Tensor &Lin, const Tensor &Rin,
                                      const std::vector<cytnx_uint64> &idxl,
                                      const std::vector<cytnx_uint64> &idxr,
                                      cutensorDataType_t type,
                                      cutensorComputeDescriptor_t descCompute, void *alpha,
                                      void *beta) {
      // hostType alpha = (hostType){1.f, 0.f};
      // hostType beta = (hostType){0.f, 0.f};

      // mode stores the label of each dimesion, which the dimension of same label would be
      // contracted label are encoded from 0 (new_label) contracted dimenstion, Tl's dimension, Tr's
      // dimension

      /*************************
       * Initialize labels for Tensors.
       * Label that exists in both Rin and Lin will be contracted.
       *************************/

      cytnx_error_msg((out.shape().size() > INT32_MAX),
                      "contraction error: dimesion exceed INT32_MAX", 0);

      std::vector<cytnx_int32> labelL(Lin.shape().size(), INT32_MAX);
      std::vector<cytnx_int32> labelR(Rin.shape().size(), INT32_MAX);
      std::vector<cytnx_int32> labelOut(Rin.shape().size() + Lin.shape().size() - 2 * idxl.size());

      cytnx_int32 new_idx;
      for (new_idx = 0; new_idx < idxl.size(); new_idx++) {
        labelL[idxl[new_idx]] = new_idx;
        labelR[idxr[new_idx]] = new_idx;
      }
      for (cytnx_int32 i = 0; i < labelL.size(); i++) {
        if (labelL[i] == INT32_MAX) {
          labelL[i] = new_idx;
          new_idx++;
        }
      }
      for (cytnx_int32 i = 0; i < labelR.size(); i++) {
        if (labelR[i] == INT32_MAX) {
          labelR[i] = new_idx;
          new_idx++;
        }
      }

      // Since the index of labels ranged from 0 to idxl.size()-1
      // The start index of non-contracted labels is idxl.size()
      for (cytnx_int32 i = 0; i < labelOut.size(); i++) {
        labelOut[i] = i + idxl.size();
      }

      void *outPtr = out._impl->storage()._impl->Mem;
      void *lPtr = Lin._impl->storage()._impl->Mem;
      void *rPtr = Rin._impl->storage()._impl->Mem;

      int nlabelOut = labelOut.size();
      int nlabelL = labelL.size();
      int nlabelR = labelR.size();

      std::vector<int64_t> extentOut(out.shape().cbegin(), out.shape().cend());
      std::vector<int64_t> extentL(Lin.shape().cbegin(), Lin.shape().cend());
      std::vector<int64_t> extentR(Rin.shape().cbegin(), Rin.shape().cend());

      // reverse the labels and extents because cuTensor is column-major by default
      std::reverse(labelL.begin(), labelL.end());
      std::reverse(labelR.begin(), labelR.end());
      std::reverse(labelOut.begin(), labelOut.end());

      std::reverse(extentL.begin(), extentL.end());
      std::reverse(extentR.begin(), extentR.end());
      std::reverse(extentOut.begin(), extentOut.end());

      /*************************
       * cuTENSOR
       *************************/

      cutensorHandle_t handle;
      HANDLE_ERROR(cutensorCreate(&handle));

      /**********************
       * Create Tensor Descriptors
       **********************/

      // This is the default alignment of cudaMalloc() and may also be the default alignment of
      // cudaMallocManaged()
      cytnx_uint64 defaultAlignment = 256;
      cutensorTensorDescriptor_t descL;
      HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descL, nlabelL, extentL.data(),
                                                  NULL, /*stride*/
                                                  type, defaultAlignment));

      cutensorTensorDescriptor_t descR;
      HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descR, nlabelR, extentR.data(),
                                                  NULL, /*stride*/
                                                  type, defaultAlignment));

      cutensorTensorDescriptor_t descOut;
      HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descOut, nlabelOut, extentOut.data(),
                                                  NULL, /*stride*/
                                                  type, defaultAlignment));

      /*******************************
       * Create Contraction Descriptor
       *******************************/

      cutensorOperationDescriptor_t desc;
      HANDLE_ERROR(
        cutensorCreateContraction(handle, &desc, descL, labelL.data(), CUTENSOR_OP_IDENTITY, descR,
                                  labelR.data(), CUTENSOR_OP_IDENTITY, descOut, labelOut.data(),
                                  CUTENSOR_OP_IDENTITY, descOut, labelOut.data(), descCompute));

      /**************************
       * Set the algorithm to use
       ***************************/

      cutensorPlanPreference_t planPref;
      cutensorCreatePlanPreference(handle, &planPref, CUTENSOR_ALGO_DEFAULT,
                                   CUTENSOR_JIT_MODE_NONE);

      /**********************
       * Query workspace
       **********************/

      uint64_t workspaceSizeEstimate = 0;
      cutensorEstimateWorkspaceSize(handle, desc, planPref, CUTENSOR_WORKSPACE_DEFAULT,
                                    &workspaceSizeEstimate);

      /**************************
       * Create Contraction Plan
       **************************/

      cutensorPlan_t plan;
      cutensorCreatePlan(handle, &plan, desc, planPref, workspaceSizeEstimate);

      uint64_t actualWorkspaceSize = 0;
      cutensorPlanGetAttribute(handle, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &actualWorkspaceSize,
                               sizeof(actualWorkspaceSize));

      void *work = nullptr;
      if (actualWorkspaceSize > 0) HANDLE_CUDA_ERROR(cudaMalloc(&work, actualWorkspaceSize));

      /**********************
       * Run
       **********************/

      cutensorContract(handle, plan, (void *)alpha, lPtr, rPtr, (void *)beta, outPtr, outPtr, work,
                       actualWorkspaceSize, /* stream */ 0);

      /*************************/

      cudaDeviceSynchronize();

      if (work) checkCudaErrors(cudaFree(work));
      // XXX: Can the OperationDescriptor be destories before running?
      HANDLE_ERROR(cutensorDestroyTensorDescriptor(descL));
      HANDLE_ERROR(cutensorDestroyTensorDescriptor(descR));
      HANDLE_ERROR(cutensorDestroyTensorDescriptor(descOut));
      HANDLE_ERROR(cutensorDestroyPlanPreference(planPref));
      HANDLE_ERROR(cutensorDestroyPlan(plan));
      HANDLE_ERROR(cutensorDestroy(handle));
    }

    void cuTensordot_internal_cd(Tensor &out, const Tensor &Lin, const Tensor &Rin,
                                 const std::vector<cytnx_uint64> &idxl,
                                 const std::vector<cytnx_uint64> &idxr) {
      typedef cuDoubleComplex hostType;
      cutensorDataType_t type = CUTENSOR_C_64F;
      cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_64F;

      hostType alpha = {1., 0.};
      hostType beta = {0., 0.};

      _cuTensordot_internal(out, Lin, Rin, idxl, idxr, type, descCompute, &alpha, &beta);
    }

    void cuTensordot_internal_cf(Tensor &out, const Tensor &Lin, const Tensor &Rin,
                                 const std::vector<cytnx_uint64> &idxl,
                                 const std::vector<cytnx_uint64> &idxr) {
      typedef cuFloatComplex hostType;
      cutensorDataType_t type = CUTENSOR_C_32F;
      cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;

      hostType alpha = {1.f, 0.f};
      hostType beta = {0.f, 0.f};

      _cuTensordot_internal(out, Lin, Rin, idxl, idxr, type, descCompute, &alpha, &beta);
    }

    void cuTensordot_internal_d(Tensor &out, const Tensor &Lin, const Tensor &Rin,
                                const std::vector<cytnx_uint64> &idxl,
                                const std::vector<cytnx_uint64> &idxr) {
      typedef double hostType;
      cutensorDataType_t type = CUTENSOR_R_64F;
      cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_64F;

      hostType alpha = 1.f;
      hostType beta = 0.f;

      _cuTensordot_internal(out, Lin, Rin, idxl, idxr, type, descCompute, &alpha, &beta);
    }

    void cuTensordot_internal_f(Tensor &out, const Tensor &Lin, const Tensor &Rin,
                                const std::vector<cytnx_uint64> &idxl,
                                const std::vector<cytnx_uint64> &idxr) {
      typedef float hostType;
      cutensorDataType_t type = CUTENSOR_R_32F;
      cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;

      hostType alpha = 1.f;
      hostType beta = 0.f;

      _cuTensordot_internal(out, Lin, Rin, idxl, idxr, type, descCompute, &alpha, &beta);
    }

  }  // namespace linalg_internal

}  // namespace cytnx
#endif
