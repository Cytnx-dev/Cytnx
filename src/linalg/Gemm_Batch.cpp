#include "linalg.hpp"

#include <variant>

#include "Generator.hpp"
#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_cpu/Gemm_Batch_internal.hpp"
  #include "backend/linalg_internal_interface.hpp"

using namespace std;

namespace cytnx {
  namespace linalg {

    void Gemm_Batch(const vector<Scalar>& alpha_array, const vector<Tensor>& a_tensors,
                    const vector<Tensor>& b_tensors, const vector<Scalar>& beta_array,
                    vector<Tensor>& c_tensors, const vector<cytnx_int64>& group_size) {
      const cytnx_int64 group_count = static_cast<cytnx_int64>(group_size.size());

      // Each group must contain at least one matrix: MKL's gemm_batch rejects
      // leading-dimension = 0 even for a group with no work, and a zero entry would also
      // leave idx == total_matrices for a trailing empty group, making the per-group
      // shape lookup below read past the end of a_tensors. Negative values are also
      // rejected here so the cast to cytnx_uint64 does not wrap.
      cytnx_uint64 total_matrices = 0;
      for (cytnx_int64 group_len : group_size) {
        cytnx_error_msg(group_len <= 0,
                        "[Gemm_Batch] error, group_size entries must be positive (>= 1)%s", "\n");
        total_matrices += static_cast<cytnx_uint64>(group_len);
      }

      // Array-size checks (unconditional; O(n_groups) and negligible vs. BLAS)
      cytnx_error_msg(a_tensors.size() != total_matrices,
                      "[Gemm_Batch] error, a_tensors.size() != total tensor count%s", "\n");
      cytnx_error_msg(b_tensors.size() != total_matrices,
                      "[Gemm_Batch] error, b_tensors.size() != total tensor count%s", "\n");
      cytnx_error_msg(c_tensors.size() != total_matrices,
                      "[Gemm_Batch] error, c_tensors.size() != total tensor count%s", "\n");
      cytnx_error_msg(alpha_array.size() != static_cast<std::size_t>(group_count),
                      "[Gemm_Batch] error, alpha_array.size() != group_count%s", "\n");
      cytnx_error_msg(beta_array.size() != static_cast<std::size_t>(group_count),
                      "[Gemm_Batch] error, beta_array.size() != group_count%s", "\n");

      if (total_matrices == 0) return;

      const int device = a_tensors[0].device();

      // Per-group scalar validation
      for (cytnx_int64 g = 0; g < group_count; g++) {
        cytnx_error_msg(alpha_array[g].dtype() == Type.Void,
                        "[Gemm_Batch] error, cannot accept Void type in alpha_array%s", "\n");
        cytnx_error_msg(beta_array[g].dtype() == Type.Void,
                        "[Gemm_Batch] error, cannot accept Void type in beta_array%s", "\n");
        cytnx_error_msg(alpha_array[g].dtype() > 4,
                        "[Gemm_Batch] alpha_array only supports (complex/real)(double/float) %s",
                        "\n");
        cytnx_error_msg(beta_array[g].dtype() > 4,
                        "[Gemm_Batch] beta_array only supports (complex/real)(double/float) %s",
                        "\n");
      }

      // Per-tensor validation
      for (cytnx_uint64 i = 0; i < total_matrices; i++) {
        cytnx_error_msg(a_tensors[i].shape().size() != 2,
                        "[Gemm_Batch] error, tensors a , Gemm can only operate on rank-2 Tensor.%s",
                        "\n");
        cytnx_error_msg(b_tensors[i].shape().size() != 2,
                        "[Gemm_Batch] error, tensors b , Gemm can only operate on rank-2 Tensor.%s",
                        "\n");
        cytnx_error_msg(c_tensors[i].shape().size() != 2,
                        "[Gemm_Batch] error, tensors c , Gemm can only operate on rank-2 Tensor.%s",
                        "\n");
        cytnx_error_msg(a_tensors[i].device() != device,
                        "[Gemm_Batch] error tensors should all on same device.%s", "\n");
        cytnx_error_msg(b_tensors[i].device() != device,
                        "[Gemm_Batch] error tensors should all on same device.%s", "\n");
        cytnx_error_msg(c_tensors[i].device() != device,
                        "[Gemm_Batch] error tensors should all on same device.%s", "\n");
        cytnx_error_msg(a_tensors[i].dtype() > 4,
                        "[Gemm_Batch] a_tensors only supports (complex/real)(double/float) %s",
                        "\n");
        cytnx_error_msg(b_tensors[i].dtype() > 4,
                        "[Gemm_Batch] b_tensors only supports (complex/real)(double/float) %s",
                        "\n");
        cytnx_error_msg(c_tensors[i].dtype() > 4,
                        "[Gemm_Batch] c_tensors only supports (complex/real)(double/float) %s",
                        "\n");
        cytnx_error_msg(a_tensors[i].dtype() == Type.Void,
                        "[Gemm_Batch] error tensor with Type.Void cannot perform arithmetic.%s",
                        "\n");
        cytnx_error_msg(b_tensors[i].dtype() == Type.Void,
                        "[Gemm_Batch] error tensor with Type.Void cannot perform arithmetic.%s",
                        "\n");
        cytnx_error_msg(c_tensors[i].dtype() == Type.Void,
                        "[Gemm_Batch] error tensor with Type.Void cannot perform arithmetic.%s",
                        "\n");
        // Dimension compatibility
        cytnx_error_msg(a_tensors[i].shape()[1] != b_tensors[i].shape()[0],
                        "[Gemm_Batch] error, a_tensors[%d],b_tensors[%d] dimension not match.%s", i,
                        i, "\n");
        cytnx_error_msg(a_tensors[i].shape()[0] != c_tensors[i].shape()[0],
                        "[Gemm_Batch] error, a_tensors[%d],c_tensors[%d] dimension not match.%s", i,
                        i, "\n");
        cytnx_error_msg(b_tensors[i].shape()[1] != c_tensors[i].shape()[1],
                        "[Gemm_Batch] error, b_tensors[%d],c_tensors[%d] dimension not match.%s", i,
                        i, "\n");
      }

      // Within-group dimension consistency (MKL needs one m/n/k per group)
      {
        cytnx_uint64 idx = 0;
        for (cytnx_int64 g = 0; g < group_count; g++) {
          const auto m0 = a_tensors[idx].shape()[0];
          const auto n0 = b_tensors[idx].shape()[1];
          const auto k0 = a_tensors[idx].shape()[1];
          for (cytnx_int64 j = 1; j < group_size[g]; j++) {
            cytnx_error_msg(
              a_tensors[idx + j].shape()[0] != m0,
              "[Gemm_Batch] error, matrices in group %d have inconsistent row count%s", (int)g,
              "\n");
            cytnx_error_msg(
              b_tensors[idx + j].shape()[1] != n0,
              "[Gemm_Batch] error, matrices in group %d have inconsistent col count%s", (int)g,
              "\n");
            cytnx_error_msg(
              a_tensors[idx + j].shape()[1] != k0,
              "[Gemm_Batch] error, matrices in group %d have inconsistent inner dim%s", (int)g,
              "\n");
          }
          idx += static_cast<cytnx_uint64>(group_size[g]);
        }
      }

      // Promoted dtype: the highest-precision type among all tensors and scalars.
      // Scalars with higher precision than the tensors promote the tensors upward so that BLAS
      // operates uniformly at the promoted precision without losing scalar bits.
      int promoted_dtype = a_tensors[0].dtype();
      for (cytnx_uint64 i = 0; i < total_matrices; i++) {
        if (a_tensors[i].dtype() < promoted_dtype) promoted_dtype = a_tensors[i].dtype();
        if (b_tensors[i].dtype() < promoted_dtype) promoted_dtype = b_tensors[i].dtype();
        if (c_tensors[i].dtype() < promoted_dtype) promoted_dtype = c_tensors[i].dtype();
      }
      for (cytnx_int64 g = 0; g < group_count; g++) {
        if (alpha_array[g].dtype() < promoted_dtype) promoted_dtype = alpha_array[g].dtype();
        if (beta_array[g].dtype() < promoted_dtype) promoted_dtype = beta_array[g].dtype();
      }

      // Promote and make contiguous
      vector<Tensor> a_promoted(a_tensors), b_promoted(b_tensors);
      vector<Scalar> alpha_promoted(alpha_array), beta_promoted(beta_array);
      for (cytnx_uint64 i = 0; i < total_matrices; i++) {
        if (a_promoted[i].dtype() != promoted_dtype)
          a_promoted[i] = a_tensors[i].astype(promoted_dtype);
        if (b_promoted[i].dtype() != promoted_dtype)
          b_promoted[i] = b_tensors[i].astype(promoted_dtype);
        if (c_tensors[i].dtype() != promoted_dtype)
          c_tensors[i] = c_tensors[i].astype(promoted_dtype);
        if (!a_promoted[i].is_contiguous()) a_promoted[i] = a_promoted[i].contiguous();
        if (!b_promoted[i].is_contiguous()) b_promoted[i] = b_promoted[i].contiguous();
        if (!c_tensors[i].is_contiguous()) c_tensors[i] = c_tensors[i].contiguous();
      }
      for (cytnx_int64 g = 0; g < group_count; g++) {
        if (alpha_array[g].dtype() != promoted_dtype)
          alpha_promoted[g] = alpha_array[g].astype(promoted_dtype);
        if (beta_array[g].dtype() != promoted_dtype)
          beta_promoted[g] = beta_array[g].astype(promoted_dtype);
      }

      // Raw data pointer arrays
      vector<void*> a_data(total_matrices), b_data(total_matrices), c_data(total_matrices);
      for (cytnx_uint64 i = 0; i < total_matrices; i++) {
        a_data[i] = a_promoted[i].storage()._impl->data();
        b_data[i] = b_promoted[i].storage()._impl->data();
        c_data[i] = c_tensors[i].storage()._impl->data();
      }

      // Per-group dimension arrays (MKL contract: one m/n/k per group, derived from first matrix)
      const auto blas_group_sizes = vec_cast<cytnx_int64, blas_int>(group_size);
      constexpr char kNoTranspose = 'N';
      vector<char> trans_flags(group_count, kNoTranspose);
      vector<blas_int> ms(group_count), ns(group_count), ks(group_count);
      {
        cytnx_uint64 idx = 0;
        for (cytnx_int64 g = 0; g < group_count; g++) {
          ms[g] = static_cast<blas_int>(a_tensors[idx].shape()[0]);
          ns[g] = static_cast<blas_int>(b_tensors[idx].shape()[1]);
          ks[g] = static_cast<blas_int>(a_tensors[idx].shape()[1]);
          idx += static_cast<cytnx_uint64>(group_size[g]);
        }
      }

      if (device == Device.cpu) {
        // Row-to-column-major swap: pass B before A, n before m.
        // Dispatch by type using std::visit on the promoted first tensor's typed pointer.
        std::visit(
          [&](auto* typed_ptr) {
            using T = std::remove_const_t<std::remove_pointer_t<decltype(typed_ptr)>>;
            if constexpr (is_complex_v<T> || std::is_floating_point_v<T>) {
              linalg_internal::Gemm_Batch<T>(
                trans_flags.data(), trans_flags.data(), ns.data(), ms.data(), ks.data(),
                alpha_promoted, (const void**)b_data.data(), ns.data(), (const void**)a_data.data(),
                ks.data(), beta_promoted, c_data.data(), ns.data(),
                static_cast<blas_int>(group_count), blas_group_sizes.data());
            } else {
              cytnx_error_msg(true, "[Gemm_Batch] unsupported dtype%s", "\n");
            }
          },
          a_promoted[0].ptr());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(device));
        linalg_internal::lii.cuGemm_Batch_ii[promoted_dtype](
          trans_flags.data(), trans_flags.data(), ns.data(), ms.data(), ks.data(), alpha_promoted,
          (const void**)b_data.data(), ns.data(), (const void**)a_data.data(), ks.data(),
          beta_promoted, c_data.data(), ns.data(), group_count, blas_group_sizes.data());
  #else
        cytnx_error_msg(true, "[Gemm_Batch] fatal error,%s",
                        "try to use GPU but not compiled with GPU support.\n");
  #endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
