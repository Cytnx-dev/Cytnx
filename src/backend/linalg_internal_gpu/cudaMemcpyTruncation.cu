#include <stdlib.h>
#include <stdio.h>

#include <vector>

#include "cudaMemcpyTruncation.hpp"

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
#endif

namespace cytnx {
  namespace linalg_internal {

#ifdef UNI_GPU
    /**
    @brief Truncate the singular values and the matrices U and vT inline
    @param[in,out] U  UniTensor with type cytnx_complex128 entries
    @param[in,out] vT UniTensor with type cytnx_complex128 entries
    @param[in,out] S  UniTensor with type cytnx_double entries
    @param[out] terr  truncated singular values, same type as S; depends on return_err
    @param[in] keepdim  number of singular values to keep at most
    @param[out] err   singular values < err are truncated
    @param[in] is_U   truncate U?
    @param[in] is_vT  truncate vT?
    @param[in] return_err return type of terr;
                          0: no error returned
                          1: only first trunceted singular value
                          2: all truncated singular values
    @param[in] is_vT  minimum number of singular values to keep
    */
    void cudaMemcpyTruncation_cd(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                                 const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                                 const bool &is_vT, const unsigned int &return_err,
                                 const unsigned int &mindim) {
      // determine the truc_dim
      cytnx_uint64 Kdim = keepdim;
      cytnx_uint64 nums = S.storage().size();
      if (nums < keepdim) {
        Kdim = nums;
      }
      cytnx_uint64 truc_dim = Kdim;
      for (cytnx_int64 i = Kdim - 1; i >= 0; i--) {
        if (((cytnx_double *)S._impl->storage()._impl->data())[i] < err and
            truc_dim - 1 >= mindim) {
          truc_dim--;
        } else {
          break;
        }
      }
      if (truc_dim == 0) {
        truc_dim = 1;
      }
      if (truc_dim != nums) {
        // perform the manual truncation

        Tensor newS = Tensor({truc_dim}, S.dtype(), S.device());
        HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_double *)newS._impl->storage()._impl->data(),
                                     (cytnx_double *)S._impl->storage()._impl->data(),
                                     truc_dim * sizeof(cytnx_double), cudaMemcpyDeviceToDevice));
        if (is_U) {
          Tensor newU = Tensor({U.shape()[0], truc_dim}, U.dtype(), U.device());

          int src = 0;
          int dest = 0;
          // copy with strides.
          for (int i = 0; i < U.shape()[0]; i++) {
            HANDLE_CUDA_ERROR(
              cudaMemcpy((cytnx_complex128 *)newU._impl->storage()._impl->data() + src,
                         (cytnx_complex128 *)U._impl->storage()._impl->data() + dest,
                         truc_dim * sizeof(cytnx_complex128), cudaMemcpyDeviceToDevice));
            src += truc_dim;
            dest += U.shape()[1];
          }
          U = newU;
        }
        if (is_vT) {
          Tensor newvT = Tensor({truc_dim, vT.shape()[1]}, vT.dtype(), vT.device());
          // simply copy a new one dropping the tail.
          HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_complex128 *)newvT._impl->storage()._impl->data(),
                                       (cytnx_complex128 *)vT._impl->storage()._impl->data(),
                                       vT.shape()[1] * truc_dim * sizeof(cytnx_complex128),
                                       cudaMemcpyDeviceToDevice));
          vT = newvT;
        }
        if (return_err == 1) {
          Tensor newterr = Tensor({1}, S.dtype(), S.device());
          ((cytnx_double *)newterr._impl->storage()._impl->data())[0] =
            ((cytnx_double *)S._impl->storage()._impl->data())[truc_dim];
          terr = newterr;
        } else if (return_err) {
          cytnx_uint64 discared_dim = S.shape()[0] - truc_dim;
          Tensor newterr = Tensor({discared_dim}, S.dtype(), S.device());
          HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_double *)newterr._impl->storage()._impl->data(),
                                       (cytnx_double *)S._impl->storage()._impl->data() + truc_dim,
                                       discared_dim * sizeof(cytnx_double),
                                       cudaMemcpyDeviceToDevice));
          terr = newterr;
        }
        S = newS;
      }
    }

    /**
    @brief Truncate the singular values and the matrices U and vT inline
    @param[in,out] U  UniTensor with type cytnx_complex64 entries
    @param[in,out] vT UniTensor with type cytnx_complex64 entries
    @param[in,out] S  UniTensor with type cytnx_float entries
    @param[out] terr  truncated singular values, same type as S; depends on return_err
    @param[in] keepdim  number of singular values to keep at most
    @param[out] err   singular values < err are truncated
    @param[in] is_U   truncate U?
    @param[in] is_vT  truncate vT?
    @param[in] return_err return type of terr;
                          0: no error returned
                          1: only first trunceted singular value
                          2: all truncated singular values
    @param[in] is_vT  minimum number of singular values to keep
    */
    void cudaMemcpyTruncation_cf(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                                 const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                                 const bool &is_vT, const unsigned int &return_err,
                                 const unsigned int &mindim) {
      // determine the truc_dim
      cytnx_uint64 Kdim = keepdim;
      cytnx_uint64 nums = S.storage().size();
      if (nums < keepdim) {
        Kdim = nums;
      }
      cytnx_uint64 truc_dim = Kdim;
      for (cytnx_int64 i = Kdim - 1; i >= 0; i--) {
        if (((cytnx_float *)S._impl->storage()._impl->data())[i] < err and truc_dim - 1 >= mindim) {
          truc_dim--;
        } else {
          break;
        }
      }
      if (truc_dim == 0) {
        truc_dim = 1;
      }
      if (truc_dim != nums) {
        // perform the manual truncation

        Tensor newS = Tensor({truc_dim}, S.dtype(), S.device());
        HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_float *)newS._impl->storage()._impl->data(),
                                     (cytnx_float *)S._impl->storage()._impl->data(),
                                     truc_dim * sizeof(cytnx_float), cudaMemcpyDeviceToDevice));
        if (is_U) {
          Tensor newU = Tensor({U.shape()[0], truc_dim}, U.dtype(), U.device());

          int src = 0;
          int dest = 0;
          // copy with strides.
          for (int i = 0; i < U.shape()[0]; i++) {
            HANDLE_CUDA_ERROR(
              cudaMemcpy((cytnx_complex64 *)newU._impl->storage()._impl->data() + src,
                         (cytnx_complex64 *)U._impl->storage()._impl->data() + dest,
                         truc_dim * sizeof(cytnx_complex64), cudaMemcpyDeviceToDevice));
            src += truc_dim;
            dest += U.shape()[1];
          }
          U = newU;
        }
        if (is_vT) {
          Tensor newvT = Tensor({truc_dim, vT.shape()[1]}, vT.dtype(), vT.device());
          // simply copy a new one dropping the tail.
          HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_complex64 *)newvT._impl->storage()._impl->data(),
                                       (cytnx_complex64 *)vT._impl->storage()._impl->data(),
                                       vT.shape()[1] * truc_dim * sizeof(cytnx_complex64),
                                       cudaMemcpyDeviceToDevice));
          vT = newvT;
        }
        if (return_err == 1) {
          Tensor newterr = Tensor({1}, S.dtype(), S.device());
          ((cytnx_float *)newterr._impl->storage()._impl->data())[0] =
            ((cytnx_float *)S._impl->storage()._impl->data())[truc_dim];
          terr = newterr;
        } else if (return_err) {
          cytnx_uint64 discared_dim = S.shape()[0] - truc_dim;
          Tensor newterr = Tensor({discared_dim}, S.dtype(), S.device());
          HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_float *)newterr._impl->storage()._impl->data(),
                                       (cytnx_float *)S._impl->storage()._impl->data() + truc_dim,
                                       discared_dim * sizeof(cytnx_float),
                                       cudaMemcpyDeviceToDevice));
          terr = newterr;
        }
        S = newS;
      }
    }

    /**
    @brief Truncate the singular values and the matrices U and vT inline
    @param[in,out] U  UniTensor with type cytnx_double entries
    @param[in,out] vT UniTensor with type cytnx_double entries
    @param[in,out] S  UniTensor with type cytnx_double entries
    @param[out] terr  truncated singular values, same type as S; depends on return_err
    @param[in] keepdim  number of singular values to keep at most
    @param[out] err   singular values < err are truncated
    @param[in] is_U   truncate U?
    @param[in] is_vT  truncate vT?
    @param[in] return_err return type of terr;
                          0: no error returned
                          1: only first trunceted singular value
                          2: all truncated singular values
    @param[in] is_vT  minimum number of singular values to keep
    */
    void cudaMemcpyTruncation_d(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                                const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                                const bool &is_vT, const unsigned int &return_err,
                                const unsigned int &mindim) {
      // determine the truc_dim
      cytnx_uint64 Kdim = keepdim;
      cytnx_uint64 nums = S.storage().size();
      if (nums < keepdim) {
        Kdim = nums;
      }
      cytnx_uint64 truc_dim = Kdim;
      for (cytnx_int64 i = Kdim - 1; i >= 0; i--) {
        if (((cytnx_double *)S._impl->storage()._impl->data())[i] < err and
            truc_dim - 1 >= mindim) {
          truc_dim--;
        } else {
          break;
        }
      }
      if (truc_dim == 0) {
        truc_dim = 1;
      }
      if (truc_dim != nums) {
        // perform the manual truncation

        Tensor newS = Tensor({truc_dim}, S.dtype(), S.device());
        HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_double *)newS._impl->storage()._impl->data(),
                                     (cytnx_double *)S._impl->storage()._impl->data(),
                                     truc_dim * sizeof(cytnx_double), cudaMemcpyDeviceToDevice));
        if (is_U) {
          Tensor newU = Tensor({U.shape()[0], truc_dim}, U.dtype(), U.device());

          int src = 0;
          int dest = 0;
          // copy with strides.
          for (int i = 0; i < U.shape()[0]; i++) {
            HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_double *)newU._impl->storage()._impl->data() + src,
                                         (cytnx_double *)U._impl->storage()._impl->data() + dest,
                                         truc_dim * sizeof(cytnx_double),
                                         cudaMemcpyDeviceToDevice));
            src += truc_dim;
            dest += U.shape()[1];
          }
          U = newU;
        }
        if (is_vT) {
          Tensor newvT = Tensor({truc_dim, vT.shape()[1]}, vT.dtype(), vT.device());
          // simply copy a new one dropping the tail.
          HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_double *)newvT._impl->storage()._impl->data(),
                                       (cytnx_double *)vT._impl->storage()._impl->data(),
                                       vT.shape()[1] * truc_dim * sizeof(cytnx_double),
                                       cudaMemcpyDeviceToDevice));
          vT = newvT;
        }
        if (return_err == 1) {
          Tensor newterr = Tensor({1}, S.dtype(), S.device());
          ((cytnx_double *)newterr._impl->storage()._impl->data())[0] =
            ((cytnx_double *)S._impl->storage()._impl->data())[truc_dim];
          terr = newterr;
        } else if (return_err) {
          cytnx_uint64 discared_dim = S.shape()[0] - truc_dim;
          Tensor newterr = Tensor({discared_dim}, S.dtype(), S.device());
          HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_double *)newterr._impl->storage()._impl->data(),
                                       (cytnx_double *)S._impl->storage()._impl->data() + truc_dim,
                                       discared_dim * sizeof(cytnx_double),
                                       cudaMemcpyDeviceToDevice));
          terr = newterr;
        }
        S = newS;
      }
    }

    /**
    @brief Truncate the singular values and the matrices U and vT inline
    @param[in,out] U  UniTensor with type cytnx_float entries
    @param[in,out] vT UniTensor with type cytnx_float entries
    @param[in,out] S  UniTensor with type cytnx_float entries
    @param[out] terr  truncated singular values, same type as S; depends on return_err
    @param[in] keepdim  number of singular values to keep at most
    @param[out] err   singular values < err are truncated
    @param[in] is_U   truncate U?
    @param[in] is_vT  truncate vT?
    @param[in] return_err return type of terr;
                          0: no error returned
                          1: only first trunceted singular value
                          2: all truncated singular values
    @param[in] is_vT  minimum number of singular values to keep
    */
    void cudaMemcpyTruncation_f(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                                const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                                const bool &is_vT, const unsigned int &return_err,
                                const unsigned int &mindim) {
      // determine the truc_dim
      cytnx_uint64 Kdim = keepdim;
      cytnx_uint64 nums = S.storage().size();
      if (nums < keepdim) {
        Kdim = nums;
      }
      cytnx_uint64 truc_dim = Kdim;
      for (cytnx_int64 i = Kdim - 1; i >= 0; i--) {
        if (((cytnx_float *)S._impl->storage()._impl->data())[i] < err and truc_dim - 1 >= mindim) {
          truc_dim--;
        } else {
          break;
        }
      }
      if (truc_dim == 0) {
        truc_dim = 1;
      }
      if (truc_dim != nums) {
        // perform the manual truncation

        Tensor newS = Tensor({truc_dim}, S.dtype(), S.device());
        HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_float *)newS._impl->storage()._impl->data(),
                                     (cytnx_float *)S._impl->storage()._impl->data(),
                                     truc_dim * sizeof(cytnx_float), cudaMemcpyDeviceToDevice));
        if (is_U) {
          Tensor newU = Tensor({U.shape()[0], truc_dim}, U.dtype(), U.device());

          int src = 0;
          int dest = 0;
          // copy with strides.
          for (int i = 0; i < U.shape()[0]; i++) {
            HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_float *)newU._impl->storage()._impl->data() + src,
                                         (cytnx_float *)U._impl->storage()._impl->data() + dest,
                                         truc_dim * sizeof(cytnx_float), cudaMemcpyDeviceToDevice));
            src += truc_dim;
            dest += U.shape()[1];
          }
          U = newU;
        }
        if (is_vT) {
          Tensor newvT = Tensor({truc_dim, vT.shape()[1]}, vT.dtype(), vT.device());
          // simply copy a new one dropping the tail.
          HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_float *)newvT._impl->storage()._impl->data(),
                                       (cytnx_float *)vT._impl->storage()._impl->data(),
                                       vT.shape()[1] * truc_dim * sizeof(cytnx_float),
                                       cudaMemcpyDeviceToDevice));
          vT = newvT;
        }
        if (return_err == 1) {
          Tensor newterr = Tensor({1}, S.dtype(), S.device());
          ((cytnx_float *)newterr._impl->storage()._impl->data())[0] =
            ((cytnx_float *)S._impl->storage()._impl->data())[truc_dim];
          terr = newterr;
        } else if (return_err) {
          cytnx_uint64 discared_dim = S.shape()[0] - truc_dim;
          Tensor newterr = Tensor({discared_dim}, S.dtype(), S.device());
          HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_float *)newterr._impl->storage()._impl->data(),
                                       (cytnx_float *)S._impl->storage()._impl->data() + truc_dim,
                                       discared_dim * sizeof(cytnx_float),
                                       cudaMemcpyDeviceToDevice));
          terr = newterr;
        }
        S = newS;
      }
    }
#endif
  }  // namespace linalg_internal
}  // namespace cytnx
