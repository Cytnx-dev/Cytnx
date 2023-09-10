#include "linalg.hpp"
#include <iostream>
#include "Tensor.hpp"
#include "Generator.hpp"

#ifdef BACKEND_TORCH
#else
  #include "../backend/linalg_internal_interface.hpp"
using namespace std;
namespace cytnx {
  namespace linalg {
    // Internal use only, not exposed to user.
    // transa, transb are not supported yet in GPU.
    void __Gemm_Batch(const vector<char> &transa_array, const vector<char> &transb_array,
                      const vector<blas_int> &m_array, const vector<blas_int> &n_array,
                      const vector<blas_int> &k_array, const vector<Scalar> &alpha_array,
                      const void **a_array, const void **b_array, const vector<Scalar> &beta_array,
                      void **c_array, const blas_int group_count,
                      const vector<blas_int> &group_size, const unsigned int dtype,
                      const int device) {
      bool check_parameters = false;
      bool guarentee_same_good_type = false;
      if (check_parameters) {
        cytnx_error_msg(group_size.size() != group_count,
                        "[Gemm_Batch] error, group_size.size() != group_count%s", "\n");
        // check tensor count
        cytnx_uint64 tot_tns = 0;
        for (cytnx_uint64 i = 0; i < group_count; i++) {
          tot_tns += group_size[i];
        }
        cytnx_error_msg(alpha_array.size() != tot_tns,
                        "[Gemm_Batch] error, alpha_array.size() != total tensor count%s", "\n");
        cytnx_error_msg(beta_array.size() != tot_tns,
                        "[Gemm_Batch] error, beta_array.size() != total tensor count%s", "\n");
      }
      if (!guarentee_same_good_type) {
        for (cytnx_uint64 i = 0; i < alpha_array.size(); i++) {
          // check Void type
          cytnx_error_msg(alpha_array[i].dtype() == Type.Void,
                          "[Gemm_Batch] error, cannot accept Void type in alpha_array%s", "\n");
          cytnx_error_msg(beta_array[i].dtype() == Type.Void,
                          "[Gemm_Batch] error, cannot accept Void type in beta_array%s", "\n");

          // check invalid type
          cytnx_error_msg(alpha_array[i].dtype() > 4,
                          "[Gemm_Batch] alpha_array only supports (complex/real)(double/float) %s",
                          "\n");
          cytnx_error_msg(beta_array[i].dtype() > 4,
                          "[Gemm_Batch] beta_array only supports (complex/real)(double/float) %s",
                          "\n");
        }
        // convert dtype:
        vector<Scalar> tmp_alpha_array(alpha_array), tmp_beta_array(beta_array);
        for (cytnx_uint64 i = 0; i < alpha_array.size(); i++) {
          if (alpha_array[i].dtype() != dtype) tmp_alpha_array[i] = alpha_array[i].astype(dtype);
          if (beta_array[i].dtype() != dtype) tmp_beta_array[i] = beta_array[i].astype(dtype);
        }
        if (device == Device.cpu) {
          linalg_internal::lii.Gemm_Batch_ii[dtype](
            transb_array.data(), transa_array.data(), n_array.data(), m_array.data(),
            k_array.data(), tmp_alpha_array, (const void **)b_array, n_array.data(),
            (const void **)a_array, k_array.data(), tmp_beta_array, c_array, n_array.data(),
            group_count, group_size.data());
        } else {
  #ifdef UNI_GPU
          checkCudaErrors(cudaSetDevice(device));
          linalg_internal::lii.cuGemm_Batch_ii[dtype](
            transb_array.data(), transa_array.data(), n_array.data(), m_array.data(),
            k_array.data(), tmp_alpha_array, (const void **)b_array, n_array.data(),
            (const void **)a_array, k_array.data(), tmp_beta_array, (void **)c_array,
            n_array.data(), group_count, group_size.data());
  #else
          cytnx_error_msg(true, "[Gemm_Batch] fatal error,%s",
                          "try to use GPU but not compiled with GPU support.\n");
  #endif
        }
      }
      //       if(device==Device.cpu){
      //         linalg_internal::lii.Gemm_Batch_ii[dtype](transb_array.data(), transa_array.data(),
      //         n_array.data(), m_array.data(), k_array.data(),
      //                 alpha_array, b_array, n_array.data(), a_array, k_array.data(),
      //                 beta_array, c_array, n_array.data(), group_count, group_size.data());
      //       }else{
      // #ifdef UNI_GPU
      //         checkCudaErrors(cudaSetDevice(device));
      //         linalg_internal::lii.cuGemm_Batch_ii[dtype](transb_array.data(),
      //         transa_array.data(), n_array.data(), m_array.data(), k_array.data(),
      //                 alpha_array, (const void**)b_array, n_array.data(), (const void**)a_array,
      //                 k_array.data(), beta_array, (void**)c_array, n_array.data(), group_count,
      //                 group_size.data());
      // #else
      //         cytnx_error_msg(true,"[Gemm_Batch] fatal error,%s","try to use GPU but not compiled
      //         with GPU support.\n");
      // #endif
      //       }
    }
    void Gemm_Batch(const vector<cytnx_int64> &m_array, const vector<cytnx_int64> &n_array,
                    const vector<cytnx_int64> &k_array, const vector<Scalar> &alpha_array,
                    const vector<Tensor> &a_tensors, const vector<Tensor> &b_tensors,
                    const vector<Scalar> &beta_array, vector<Tensor> &c_tensors,
                    const cytnx_int64 group_count, const vector<cytnx_int64> &group_size) {
      std::vector<char> transs(a_tensors.size(), 'N');

      if (User_debug) {
        cytnx_error_msg(group_size.size() != group_count,
                        "[Gemm_Batch] error, group_size.size() != group_count%s", "\n");
        // check tensor count
        cytnx_uint64 tot_tns = 0;
        for (cytnx_uint64 i = 0; i < group_count; i++) {
          tot_tns += group_size[i];
        }
        cytnx_error_msg(a_tensors.size() != tot_tns,
                        "[Gemm_Batch] error, a_tensors.size() != total tensor count%s", "\n");
        cytnx_error_msg(b_tensors.size() != tot_tns,
                        "[Gemm_Batch] error, b_tensors.size() != total tensor count%s", "\n");
        cytnx_error_msg(c_tensors.size() != tot_tns,
                        "[Gemm_Batch] error, c_tensors.size() != total tensor count%s", "\n");
        cytnx_error_msg(alpha_array.size() != tot_tns,
                        "[Gemm_Batch] error, alpha_array.size() != total tensor count%s", "\n");
        cytnx_error_msg(beta_array.size() != tot_tns,
                        "[Gemm_Batch] error, beta_array.size() != total tensor count%s", "\n");

        for (Tensor t : a_tensors) {
          cytnx_error_msg(
            t.shape().size() != 2,
            "[Gemm_Batch] error, tensors a , Gemm can only operate on rank-2 Tensor.%s", "\n");
        }
        for (Tensor t : b_tensors) {
          cytnx_error_msg(
            t.shape().size() != 2,
            "[Gemm_Batch] error, tensors b , Gemm can only operate on rank-2 Tensor.%s", "\n");
        }
        for (Tensor t : c_tensors) {
          cytnx_error_msg(
            t.shape().size() != 2,
            "[Gemm_Batch] error, tensors c , Gemm can only operate on rank-2 Tensor.%s", "\n");
        }
        int tmp_dev = a_tensors[0].device();
        for (Tensor t : a_tensors) {
          cytnx_error_msg(t.device() != tmp_dev,
                          "[Gemm_Batch] error tensors should all on same device.%s", "\n");
        }
        for (Tensor t : b_tensors) {
          cytnx_error_msg(t.device() != tmp_dev,
                          "[Gemm_Batch] error tensors should all on same device.%s", "\n");
        }
        for (Tensor t : c_tensors) {
          cytnx_error_msg(t.device() != tmp_dev,
                          "[Gemm_Batch] error tensors should all on same device.%s", "\n");
        }

        // check dimension match
        for (unsigned int i = 0; i < a_tensors.size(); i++) {
          cytnx_error_msg(a_tensors[i].shape()[1] != b_tensors[i].shape()[0],
                          "[Gemm_Batch] error, a_tensors[%d],b_tensors[%d] dimension not match.%s",
                          i, i, "\n");
          cytnx_error_msg(a_tensors[i].shape()[0] != c_tensors[i].shape()[0],
                          "[Gemm_Batch] error, a_tensors[%d],c_tensors[%d] dimension not match.%s",
                          i, i, "\n");
          cytnx_error_msg(b_tensors[i].shape()[1] != c_tensors[i].shape()[1],
                          "[Gemm_Batch] error, b_tensors[%d],c_tensors[%d] dimension not match.%s",
                          i, i, "\n");
        }

        // contiguous?
        for (cytnx_uint64 i = 0; i < a_tensors.size(); i++) {
          cytnx_error_msg(!a_tensors[i].is_contiguous(),
                          "[Gemm_Batch] error tensor a_tensors[%d] should be contiguous.%s", i,
                          "\n");
          cytnx_error_msg(!b_tensors[i].is_contiguous(),
                          "[Gemm_Batch] error tensor b_tensors[%d] should be contiguous.%s", i,
                          "\n");
          cytnx_error_msg(!c_tensors[i].is_contiguous(),
                          "[Gemm_Batch] error tensor c_tensors[%d] should be contiguous.%s", i,
                          "\n");
        }
      }

      // checking the largest dtype!
      int fin_dtype = a_tensors[0].dtype();
      for (cytnx_uint64 i = 0; i < a_tensors.size(); i++) {
        if (a_tensors[i].dtype() < fin_dtype) fin_dtype = a_tensors[i].dtype();
        if (b_tensors[i].dtype() < fin_dtype) fin_dtype = b_tensors[i].dtype();
        if (c_tensors[i].dtype() < fin_dtype) fin_dtype = c_tensors[i].dtype();
        if (alpha_array[i].dtype() < fin_dtype) fin_dtype = alpha_array[i].dtype();
        if (beta_array[i].dtype() < fin_dtype) fin_dtype = beta_array[i].dtype();
        // check invalid type
        cytnx_error_msg(alpha_array[i].dtype() > 4,
                        "[Gemm_Batch] alpha_array only supports (complex/real)(double/float) %s",
                        "\n");
        cytnx_error_msg(beta_array[i].dtype() > 4,
                        "[Gemm_Batch] beta_array only supports (complex/real)(double/float) %s",
                        "\n");
        cytnx_error_msg(a_tensors[i].dtype() > 4,
                        "[Gemm_Batch] a_tensors only supports (complex/real)(double/float) %s",
                        "\n");
        cytnx_error_msg(b_tensors[i].dtype() > 4,
                        "[Gemm_Batch] b_tensors only supports (complex/real)(double/float) %s",
                        "\n");
        cytnx_error_msg(c_tensors[i].dtype() > 4,
                        "[Gemm_Batch] c_tensors only supports (complex/real)(double/float) %s",
                        "\n");
        // check Void type
        cytnx_error_msg(a_tensors[i].dtype() == Type.Void,
                        "[Gemm_Batch] error tensor with Type.Void cannot perform arithmetic.%s",
                        "\n");
        cytnx_error_msg(b_tensors[i].dtype() == Type.Void,
                        "[Gemm_Batch] error tensor with Type.Void cannot perform arithmetic.%s",
                        "\n");
        cytnx_error_msg(c_tensors[i].dtype() == Type.Void,
                        "[Gemm_Batch] error tensor with Type.Void cannot perform arithmetic.%s",
                        "\n");
        cytnx_error_msg(alpha_array[i].dtype() == Type.Void,
                        "[Gemm_Batch] error tensor with Type.Void cannot perform arithmetic.%s",
                        "\n");
        cytnx_error_msg(beta_array[i].dtype() == Type.Void,
                        "[Gemm_Batch] error tensor with Type.Void cannot perform arithmetic.%s",
                        "\n");
      }
      // convert dtype:
      vector<Tensor> tmp_a_tensors(a_tensors), tmp_b_tensors(b_tensors);
      vector<Scalar> tmp_alpha_array(alpha_array), tmp_beta_array(beta_array);
      for (cytnx_uint64 i = 0; i < a_tensors.size(); i++) {
        if (a_tensors[i].dtype() != fin_dtype) tmp_a_tensors[i] = a_tensors[i].astype(fin_dtype);
        if (b_tensors[i].dtype() != fin_dtype) tmp_b_tensors[i] = b_tensors[i].astype(fin_dtype);
        if (c_tensors[i].dtype() != fin_dtype) c_tensors[i] = c_tensors[i].astype(fin_dtype);
        if (alpha_array[i].dtype() != fin_dtype)
          tmp_alpha_array[i] = alpha_array[i].astype(fin_dtype);
        if (beta_array[i].dtype() != fin_dtype) tmp_beta_array[i] = beta_array[i].astype(fin_dtype);
      }
      // contiguous?
      for (cytnx_uint64 i = 0; i < a_tensors.size(); i++) {
        if (!tmp_a_tensors[i].is_contiguous()) tmp_a_tensors[i] = tmp_a_tensors[i].contiguous();
        if (!tmp_b_tensors[i].is_contiguous()) tmp_b_tensors[i] = tmp_b_tensors[i].contiguous();
        if (!c_tensors[i].is_contiguous()) c_tensors[i] = c_tensors[i].contiguous();
      }

      void *a_array[tmp_a_tensors.size()], *b_array[tmp_b_tensors.size()],
        *c_array[c_tensors.size()];
      for (cytnx_uint64 i = 0; i < a_tensors.size(); i++) {
        a_array[i] = tmp_a_tensors[i].storage()._impl->Mem;
        b_array[i] = tmp_b_tensors[i].storage()._impl->Mem;
        c_array[i] = c_tensors[i].storage()._impl->Mem;
      }
      vector<blas_int> ms(vec_cast<cytnx_int64, blas_int>(m_array)),
        ns(vec_cast<cytnx_int64, blas_int>(n_array)), ks(vec_cast<cytnx_int64, blas_int>(k_array));

      if (a_tensors[0].device() == Device.cpu) {
        linalg_internal::lii.Gemm_Batch_ii[fin_dtype](
          transs.data(), transs.data(), ns.data(), ms.data(), ks.data(), tmp_alpha_array,
          (const void **)b_array, ns.data(), (const void **)a_array, ks.data(), tmp_beta_array,
          (void **)c_array, ns.data(), (blas_int)group_count,
          vec_cast<cytnx_int64, blas_int>(group_size).data());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(a_tensors[0].device()));
        linalg_internal::lii.cuGemm_Batch_ii[fin_dtype](
          transs.data(), transs.data(), ns.data(), ms.data(), ks.data(), tmp_alpha_array,
          (const void **)b_array, ns.data(), (const void **)a_array, ks.data(), tmp_beta_array,
          (void **)c_array, ns.data(), group_count,
          vec_cast<cytnx_int64, blas_int>(group_size).data());
  #else
        cytnx_error_msg(true, "[Gemm_Batch] fatal error,%s",
                        "try to use GPU but not compiled with GPU support.\n");
  #endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
