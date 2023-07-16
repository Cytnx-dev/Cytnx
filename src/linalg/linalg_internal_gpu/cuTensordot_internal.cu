#ifdef UNI_CUTENSOR
#include "cuTensordot_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "lapack_wrapper.hpp"
#include <cutensor.h>

#define HANDLE_ERROR(x)                                                        \
  {                                                                            \
    const cutensorStatus_t err = x;                                            \
    if (err != CUTENSOR_STATUS_SUCCESS) {                                      \
      printf("Error in line %d: %s\n", __LINE__, cutensorGetErrorString(err)); \
      exit(-1);                                                                \
    }                                                                          \
  };

namespace cytnx {

  namespace linalg_internal {

    inline void _cuTensordot_internal(Tensor &out, const Tensor &Lin, const Tensor &Rin,
                                      const std::vector<cytnx_uint64> &idxl,
                                      const std::vector<cytnx_uint64> &idxr, cudaDataType_t type,
                                      cutensorComputeType_t typeCompute, void *alpha, void *beta) {
      // hostType alpha = (hostType){1.f, 0.f};
      // hostType beta = (hostType){0.f, 0.f};

      // mode stores the label of each dimesion, which the dimension of same label would be
      // contracted label are encoded from 0 (new_label) contracted dimenstion, Tl's dimension, Tr's
      // dimension

      cytnx_error_msg((out.shape().size() > INT32_MAX),
                      "contraction error: dimesion exceed INT32_MAX", 0);

      // "mode" is label
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

      // range(idxl.size(), idxl.size()+out.shape().size())
      // no vec_range fn with cytnx::int32 type, so I use the naive ways.
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

      /*************************
       * cuTENSOR
       *************************/

      cutensorHandle_t *handle;
      HANDLE_ERROR(cutensorCreate(&handle));

      /**********************
       * Create Tensor Descriptors
       **********************/

      cutensorTensorDescriptor_t descL;
      HANDLE_ERROR(cutensorInitTensorDescriptor(handle, &descL, nlabelL, extentL.data(),
                                                NULL, /*stride*/
                                                type, CUTENSOR_OP_IDENTITY));

      cutensorTensorDescriptor_t descR;
      HANDLE_ERROR(cutensorInitTensorDescriptor(handle, &descR, nlabelR, extentR.data(),
                                                NULL, /*stride*/
                                                type, CUTENSOR_OP_IDENTITY));

      cutensorTensorDescriptor_t descOut;
      HANDLE_ERROR(cutensorInitTensorDescriptor(handle, &descOut, nlabelOut, extentOut.data(),
                                                NULL, /*stride*/
                                                type, CUTENSOR_OP_IDENTITY));

      /**********************************************
       * Retrieve the memory alignment for each tensor
       **********************************************/

      uint32_t alignmentRequirementL;
      HANDLE_ERROR(cutensorGetAlignmentRequirement(handle, lPtr, &descL, &alignmentRequirementL));

      uint32_t alignmentRequirementR;
      HANDLE_ERROR(cutensorGetAlignmentRequirement(handle, rPtr, &descR, &alignmentRequirementR));

      uint32_t alignmentRequirementOut;
      HANDLE_ERROR(
        cutensorGetAlignmentRequirement(handle, outPtr, &descOut, &alignmentRequirementOut));

      /*******************************
       * Create Contraction Descriptor
       *******************************/

      cutensorContractionDescriptor_t desc;
      HANDLE_ERROR(cutensorInitContractionDescriptor(
        handle, &desc, &descL, labelL.data(), alignmentRequirementL, &descR, labelR.data(),
        alignmentRequirementR, &descOut, labelOut.data(), alignmentRequirementOut, &descOut,
        labelOut.data(), alignmentRequirementOut, typeCompute));

      /**************************
       * Set the algorithm to use
       ***************************/

      cutensorContractionFind_t find;
      HANDLE_ERROR(cutensorInitContractionFind(handle, &find, CUTENSOR_ALGO_DEFAULT));

      /**********************
       * Query workspace
       **********************/

      uint64_t worksize = 0;
      HANDLE_ERROR(cutensorContractionGetWorkspaceSize(handle, &desc, &find,
                                                       CUTENSOR_WORKSPACE_RECOMMENDED, &worksize));

      void *work = nullptr;
      if (worksize > 0) {
        if (cudaSuccess != cudaMalloc(&work, worksize)) {
          work = nullptr;
          worksize = 0;
        }
      }

      /**************************
       * Create Contraction Plan
       **************************/

      cutensorContractionPlan_t plan;
      HANDLE_ERROR(cutensorInitContractionPlan(handle, &plan, &desc, &find, worksize));

      /**********************
       * Run
       **********************/

      HANDLE_ERROR(cutensorContraction(handle, &plan, alpha, lPtr, rPtr, beta, outPtr, outPtr, work,
                                       worksize, 0 /* stream */));

      /*************************/

      if (work) checkCudaErrors(cudaFree(work));
      HANDLE_ERROR(cutensorDestroy(handle));
    }

    void cuTensordot_internal_cd(Tensor &out, const Tensor &Lin, const Tensor &Rin,
                                 const std::vector<cytnx_uint64> &idxl,
                                 const std::vector<cytnx_uint64> &idxr) {
      typedef cuDoubleComplex hostType;
      cudaDataType_t type = CUDA_C_64F;
      cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_64F;

      hostType alpha = {1., 0.};
      hostType beta = {0., 0.};

      _cuTensordot_internal(out, Lin, Rin, idxl, idxr, type, typeCompute, &alpha, &beta);
    }

    void cuTensordot_internal_cf(Tensor &out, const Tensor &Lin, const Tensor &Rin,
                                 const std::vector<cytnx_uint64> &idxl,
                                 const std::vector<cytnx_uint64> &idxr) {
      typedef cuFloatComplex hostType;
      cudaDataType_t type = CUDA_C_32F;
      cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

      hostType alpha = {1.f, 0.f};
      hostType beta = {0.f, 0.f};

      _cuTensordot_internal(out, Lin, Rin, idxl, idxr, type, typeCompute, &alpha, &beta);
    }

    void cuTensordot_internal_d(Tensor &out, const Tensor &Lin, const Tensor &Rin,
                                const std::vector<cytnx_uint64> &idxl,
                                const std::vector<cytnx_uint64> &idxr) {
      typedef double hostType;
      cudaDataType_t type = CUDA_C_64F;
      cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_64F;

      hostType alpha = 1.f;
      hostType beta = 0.f;

      _cuTensordot_internal(out, Lin, Rin, idxl, idxr, type, typeCompute, &alpha, &beta);
    }

    void cuTensordot_internal_f(Tensor &out, const Tensor &Lin, const Tensor &Rin,
                                const std::vector<cytnx_uint64> &idxl,
                                const std::vector<cytnx_uint64> &idxr) {
      typedef float hostType;
      cudaDataType_t type = CUDA_C_32F;
      cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

      hostType alpha = 1.f;
      hostType beta = 0.f;

      _cuTensordot_internal(out, Lin, Rin, idxl, idxr, type, typeCompute, &alpha, &beta);
    }

    void cuTensordot_internal_u32(Tensor &out, const Tensor &Lin, const Tensor &Rin,
                                  const std::vector<cytnx_uint64> &idxl,
                                  const std::vector<cytnx_uint64> &idxr) {
      typedef cytnx_uint32 hostType;
      cudaDataType_t type = CUDA_R_32U;

      cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32U;

      hostType alpha = 1;
      hostType beta = 0;

      _cuTensordot_internal(out, Lin, Rin, idxl, idxr, type, typeCompute, &alpha, &beta);
    }

    void cuTensordot_internal_i32(Tensor &out, const Tensor &Lin, const Tensor &Rin,
                                  const std::vector<cytnx_uint64> &idxl,
                                  const std::vector<cytnx_uint64> &idxr) {
      typedef cytnx_int32 hostType;
      cudaDataType_t type = CUDA_R_32I;

      cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32I;

      hostType alpha = 1;
      hostType beta = 0;

      _cuTensordot_internal(out, Lin, Rin, idxl, idxr, type, typeCompute, &alpha, &beta);
    }

  }  // namespace linalg_internal

}  // namespace cytnx
#endif