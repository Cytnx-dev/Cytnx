#include "memcpyTruncation.hpp"

#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "Tensor.hpp"

namespace cytnx {
  namespace linalg_internal {

    void memcpyTruncation_cd(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
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
        if (((cytnx_double *)S._impl->storage()._impl->Mem)[i] < err and truc_dim - 1 >= mindim) {
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
        memcpy((cytnx_double *)newS._impl->storage()._impl->Mem,
               (cytnx_double *)S._impl->storage()._impl->Mem, truc_dim * sizeof(cytnx_double));
        if (is_U) {
          Tensor newU = Tensor({U.shape()[0], truc_dim}, U.dtype(), U.device());

          int src = 0;
          int dest = 0;
          // copy with strides.
          for (int i = 0; i < U.shape()[0]; i++) {
            memcpy((cytnx_complex128 *)newU._impl->storage()._impl->Mem + src,
                   (cytnx_complex128 *)U._impl->storage()._impl->Mem + dest,
                   truc_dim * sizeof(cytnx_complex128));
            src += truc_dim;
            dest += U.shape()[1];
          }
          U = newU;
        }
        if (is_vT) {
          Tensor newvT = Tensor({truc_dim, vT.shape()[1]}, vT.dtype(), vT.device());
          // simply copy a new one dropping the tail.
          memcpy((cytnx_complex128 *)newvT._impl->storage()._impl->Mem,
                 (cytnx_complex128 *)vT._impl->storage()._impl->Mem,
                 vT.shape()[1] * truc_dim * sizeof(cytnx_complex128));
          vT = newvT;
        }
        if (return_err == 1) {
          Tensor newterr = Tensor({1}, S.dtype(), S.device());
          ((cytnx_double *)newterr._impl->storage()._impl->Mem)[0] =
            ((cytnx_double *)S._impl->storage()._impl->Mem)[truc_dim];
          terr = newterr;
        } else if (return_err) {
          cytnx_uint64 discared_dim = S.shape()[0] - truc_dim;
          Tensor newterr = Tensor({discared_dim}, S.dtype(), S.device());
          memcpy((cytnx_double *)newterr._impl->storage()._impl->Mem,
                 (cytnx_double *)S._impl->storage()._impl->Mem + truc_dim,
                 discared_dim * sizeof(cytnx_double));
          terr = newterr;
        }
        S = newS;
      }
    }

    void memcpyTruncation_cf(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
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
        if (((cytnx_double *)S._impl->storage()._impl->Mem)[i] < err and truc_dim - 1 >= mindim) {
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
        memcpy((cytnx_double *)newS._impl->storage()._impl->Mem,
               (cytnx_double *)S._impl->storage()._impl->Mem, truc_dim * sizeof(cytnx_double));
        if (is_U) {
          Tensor newU = Tensor({U.shape()[0], truc_dim}, U.dtype(), U.device());

          int src = 0;
          int dest = 0;
          // copy with strides.
          for (int i = 0; i < U.shape()[0]; i++) {
            memcpy((cytnx_complex64 *)newU._impl->storage()._impl->Mem + src,
                   (cytnx_complex64 *)U._impl->storage()._impl->Mem + dest,
                   truc_dim * sizeof(cytnx_complex64));
            src += truc_dim;
            dest += U.shape()[1];
          }
          U = newU;
        }
        if (is_vT) {
          Tensor newvT = Tensor({truc_dim, vT.shape()[1]}, vT.dtype(), vT.device());
          // simply copy a new one dropping the tail.
          memcpy((cytnx_complex64 *)newvT._impl->storage()._impl->Mem,
                 (cytnx_complex64 *)vT._impl->storage()._impl->Mem,
                 vT.shape()[1] * truc_dim * sizeof(cytnx_complex64));
          vT = newvT;
        }
        if (return_err == 1) {
          Tensor newterr = Tensor({1}, S.dtype(), S.device());
          ((cytnx_double *)newterr._impl->storage()._impl->Mem)[0] =
            ((cytnx_double *)S._impl->storage()._impl->Mem)[truc_dim];
          terr = newterr;
        } else if (return_err) {
          cytnx_uint64 discared_dim = S.shape()[0] - truc_dim;
          Tensor newterr = Tensor({discared_dim}, S.dtype(), S.device());
          memcpy((cytnx_double *)newterr._impl->storage()._impl->Mem,
                 (cytnx_double *)S._impl->storage()._impl->Mem + truc_dim,
                 discared_dim * sizeof(cytnx_double));
          terr = newterr;
        }
        S = newS;
      }
    }

    void memcpyTruncation_d(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
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
        if (((cytnx_double *)S._impl->storage()._impl->Mem)[i] < err and truc_dim - 1 >= mindim) {
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
        memcpy((cytnx_double *)newS._impl->storage()._impl->Mem,
               (cytnx_double *)S._impl->storage()._impl->Mem, truc_dim * sizeof(cytnx_double));
        if (is_U) {
          Tensor newU = Tensor({U.shape()[0], truc_dim}, U.dtype(), U.device());

          int src = 0;
          int dest = 0;
          // copy with strides.
          for (int i = 0; i < U.shape()[0]; i++) {
            memcpy((cytnx_double *)newU._impl->storage()._impl->Mem + src,
                   (cytnx_double *)U._impl->storage()._impl->Mem + dest,
                   truc_dim * sizeof(cytnx_double));
            src += truc_dim;
            dest += U.shape()[1];
          }
          U = newU;
        }
        if (is_vT) {
          Tensor newvT = Tensor({truc_dim, vT.shape()[1]}, vT.dtype(), vT.device());
          // simply copy a new one dropping the tail.
          memcpy((cytnx_double *)newvT._impl->storage()._impl->Mem,
                 (cytnx_double *)vT._impl->storage()._impl->Mem,
                 vT.shape()[1] * truc_dim * sizeof(cytnx_double));
          vT = newvT;
        }
        if (return_err == 1) {
          Tensor newterr = Tensor({1}, S.dtype(), S.device());
          ((cytnx_double *)newterr._impl->storage()._impl->Mem)[0] =
            ((cytnx_double *)S._impl->storage()._impl->Mem)[truc_dim];
          terr = newterr;
        } else if (return_err) {
          cytnx_uint64 discared_dim = S.shape()[0] - truc_dim;
          Tensor newterr = Tensor({discared_dim}, S.dtype(), S.device());
          memcpy((cytnx_double *)newterr._impl->storage()._impl->Mem,
                 (cytnx_double *)S._impl->storage()._impl->Mem + truc_dim,
                 discared_dim * sizeof(cytnx_double));
          terr = newterr;
        }
        S = newS;
      }
    }

    void memcpyTruncation_f(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
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
        if (((cytnx_double *)S._impl->storage()._impl->Mem)[i] < err and truc_dim - 1 >= mindim) {
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
        memcpy((cytnx_double *)newS._impl->storage()._impl->Mem,
               (cytnx_double *)S._impl->storage()._impl->Mem, truc_dim * sizeof(cytnx_double));
        if (is_U) {
          Tensor newU = Tensor({U.shape()[0], truc_dim}, U.dtype(), U.device());

          int src = 0;
          int dest = 0;
          // copy with strides.
          for (int i = 0; i < U.shape()[0]; i++) {
            memcpy((cytnx_float *)newU._impl->storage()._impl->Mem + src,
                   (cytnx_float *)U._impl->storage()._impl->Mem + dest,
                   truc_dim * sizeof(cytnx_float));
            src += truc_dim;
            dest += U.shape()[1];
          }
          U = newU;
        }
        if (is_vT) {
          Tensor newvT = Tensor({truc_dim, vT.shape()[1]}, vT.dtype(), vT.device());
          // simply copy a new one dropping the tail.
          memcpy((cytnx_float *)newvT._impl->storage()._impl->Mem,
                 (cytnx_float *)vT._impl->storage()._impl->Mem,
                 vT.shape()[1] * truc_dim * sizeof(cytnx_float));
          vT = newvT;
        }
        if (return_err == 1) {
          Tensor newterr = Tensor({1}, S.dtype(), S.device());
          ((cytnx_double *)newterr._impl->storage()._impl->Mem)[0] =
            ((cytnx_double *)S._impl->storage()._impl->Mem)[truc_dim];
          terr = newterr;
        } else if (return_err) {
          cytnx_uint64 discared_dim = S.shape()[0] - truc_dim;
          Tensor newterr = Tensor({discared_dim}, S.dtype(), S.device());
          memcpy((cytnx_double *)newterr._impl->storage()._impl->Mem,
                 (cytnx_double *)S._impl->storage()._impl->Mem + truc_dim,
                 discared_dim * sizeof(cytnx_double));
          terr = newterr;
        }
        S = newS;
      }
    }

  }  // namespace linalg_internal
}  // namespace cytnx
