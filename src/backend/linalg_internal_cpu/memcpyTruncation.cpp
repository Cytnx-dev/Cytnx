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
      // determine the trunc_dim
      cytnx_uint64 Kdim = keepdim;
      cytnx_uint64 nums = S.storage().size();
      if (nums < keepdim) {
        Kdim = nums;
      }
      cytnx_uint64 trunc_dim = Kdim;
      for (cytnx_int64 i = Kdim - 1; i >= 0; i--) {
        if (((cytnx_double *)S._impl->storage()._impl->data())[i] < err and
            trunc_dim - 1 >= mindim) {
          trunc_dim--;
        } else {
          break;
        }
      }
      if (trunc_dim == 0) {
        trunc_dim = 1;
      }
      if (trunc_dim != nums) {
        // perform the manual truncation

        Tensor newS = Tensor({trunc_dim}, S.dtype(), S.device());
        memcpy((cytnx_double *)newS._impl->storage()._impl->data(),
               (cytnx_double *)S._impl->storage()._impl->data(), trunc_dim * sizeof(cytnx_double));
        if (is_U) {
          Tensor newU = Tensor({U.shape()[0], trunc_dim}, U.dtype(), U.device());

          int src = 0;
          int dest = 0;
          // copy with strides.
          for (int i = 0; i < U.shape()[0]; i++) {
            memcpy((cytnx_complex128 *)newU._impl->storage()._impl->data() + src,
                   (cytnx_complex128 *)U._impl->storage()._impl->data() + dest,
                   trunc_dim * sizeof(cytnx_complex128));
            src += trunc_dim;
            dest += U.shape()[1];
          }
          U = newU;
        }
        if (is_vT) {
          Tensor newvT = Tensor({trunc_dim, vT.shape()[1]}, vT.dtype(), vT.device());
          // simply copy a new one dropping the tail.
          memcpy((cytnx_complex128 *)newvT._impl->storage()._impl->data(),
                 (cytnx_complex128 *)vT._impl->storage()._impl->data(),
                 vT.shape()[1] * trunc_dim * sizeof(cytnx_complex128));
          vT = newvT;
        }
        if (return_err == 1) {
          Tensor newterr = Tensor({1}, S.dtype(), S.device());
          ((cytnx_double *)newterr._impl->storage()._impl->data())[0] =
            ((cytnx_double *)S._impl->storage()._impl->data())[trunc_dim];
          terr = newterr;
        } else if (return_err) {
          cytnx_uint64 discared_dim = S.shape()[0] - trunc_dim;
          Tensor newterr = Tensor({discared_dim}, S.dtype(), S.device());
          memcpy((cytnx_double *)newterr._impl->storage()._impl->data(),
                 (cytnx_double *)S._impl->storage()._impl->data() + trunc_dim,
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
      // determine the trunc_dim
      cytnx_uint64 Kdim = keepdim;
      cytnx_uint64 nums = S.storage().size();
      if (nums < keepdim) {
        Kdim = nums;
      }
      cytnx_uint64 trunc_dim = Kdim;
      for (cytnx_int64 i = Kdim - 1; i >= 0; i--) {
        if (((cytnx_double *)S._impl->storage()._impl->data())[i] < err and
            trunc_dim - 1 >= mindim) {
          trunc_dim--;
        } else {
          break;
        }
      }
      if (trunc_dim == 0) {
        trunc_dim = 1;
      }
      if (trunc_dim != nums) {
        // perform the manual truncation

        Tensor newS = Tensor({trunc_dim}, S.dtype(), S.device());
        memcpy((cytnx_double *)newS._impl->storage()._impl->data(),
               (cytnx_double *)S._impl->storage()._impl->data(), trunc_dim * sizeof(cytnx_double));
        if (is_U) {
          Tensor newU = Tensor({U.shape()[0], trunc_dim}, U.dtype(), U.device());

          int src = 0;
          int dest = 0;
          // copy with strides.
          for (int i = 0; i < U.shape()[0]; i++) {
            memcpy((cytnx_complex64 *)newU._impl->storage()._impl->data() + src,
                   (cytnx_complex64 *)U._impl->storage()._impl->data() + dest,
                   trunc_dim * sizeof(cytnx_complex64));
            src += trunc_dim;
            dest += U.shape()[1];
          }
          U = newU;
        }
        if (is_vT) {
          Tensor newvT = Tensor({trunc_dim, vT.shape()[1]}, vT.dtype(), vT.device());
          // simply copy a new one dropping the tail.
          memcpy((cytnx_complex64 *)newvT._impl->storage()._impl->data(),
                 (cytnx_complex64 *)vT._impl->storage()._impl->data(),
                 vT.shape()[1] * trunc_dim * sizeof(cytnx_complex64));
          vT = newvT;
        }
        if (return_err == 1) {
          Tensor newterr = Tensor({1}, S.dtype(), S.device());
          ((cytnx_double *)newterr._impl->storage()._impl->data())[0] =
            ((cytnx_double *)S._impl->storage()._impl->data())[trunc_dim];
          terr = newterr;
        } else if (return_err) {
          cytnx_uint64 discared_dim = S.shape()[0] - trunc_dim;
          Tensor newterr = Tensor({discared_dim}, S.dtype(), S.device());
          memcpy((cytnx_double *)newterr._impl->storage()._impl->data(),
                 (cytnx_double *)S._impl->storage()._impl->data() + trunc_dim,
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
      // determine the trunc_dim
      cytnx_uint64 Kdim = keepdim;
      cytnx_uint64 nums = S.storage().size();
      if (nums < keepdim) {
        Kdim = nums;
      }
      cytnx_uint64 trunc_dim = Kdim;
      for (cytnx_int64 i = Kdim - 1; i >= 0; i--) {
        if (((cytnx_double *)S._impl->storage()._impl->data())[i] < err and
            trunc_dim - 1 >= mindim) {
          trunc_dim--;
        } else {
          break;
        }
      }
      if (trunc_dim == 0) {
        trunc_dim = 1;
      }
      if (trunc_dim != nums) {
        // perform the manual truncation

        Tensor newS = Tensor({trunc_dim}, S.dtype(), S.device());
        memcpy((cytnx_double *)newS._impl->storage()._impl->data(),
               (cytnx_double *)S._impl->storage()._impl->data(), trunc_dim * sizeof(cytnx_double));
        if (is_U) {
          Tensor newU = Tensor({U.shape()[0], trunc_dim}, U.dtype(), U.device());

          int src = 0;
          int dest = 0;
          // copy with strides.
          for (int i = 0; i < U.shape()[0]; i++) {
            memcpy((cytnx_double *)newU._impl->storage()._impl->data() + src,
                   (cytnx_double *)U._impl->storage()._impl->data() + dest,
                   trunc_dim * sizeof(cytnx_double));
            src += trunc_dim;
            dest += U.shape()[1];
          }
          U = newU;
        }
        if (is_vT) {
          Tensor newvT = Tensor({trunc_dim, vT.shape()[1]}, vT.dtype(), vT.device());
          // simply copy a new one dropping the tail.
          memcpy((cytnx_double *)newvT._impl->storage()._impl->data(),
                 (cytnx_double *)vT._impl->storage()._impl->data(),
                 vT.shape()[1] * trunc_dim * sizeof(cytnx_double));
          vT = newvT;
        }
        if (return_err == 1) {
          Tensor newterr = Tensor({1}, S.dtype(), S.device());
          ((cytnx_double *)newterr._impl->storage()._impl->data())[0] =
            ((cytnx_double *)S._impl->storage()._impl->data())[trunc_dim];
          terr = newterr;
        } else if (return_err) {
          cytnx_uint64 discared_dim = S.shape()[0] - trunc_dim;
          Tensor newterr = Tensor({discared_dim}, S.dtype(), S.device());
          memcpy((cytnx_double *)newterr._impl->storage()._impl->data(),
                 (cytnx_double *)S._impl->storage()._impl->data() + trunc_dim,
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
      // determine the trunc_dim
      cytnx_uint64 Kdim = keepdim;
      cytnx_uint64 nums = S.storage().size();
      if (nums < keepdim) {
        Kdim = nums;
      }
      cytnx_uint64 trunc_dim = Kdim;
      for (cytnx_int64 i = Kdim - 1; i >= 0; i--) {
        if (((cytnx_double *)S._impl->storage()._impl->data())[i] < err and
            trunc_dim - 1 >= mindim) {
          trunc_dim--;
        } else {
          break;
        }
      }
      if (trunc_dim == 0) {
        trunc_dim = 1;
      }
      if (trunc_dim != nums) {
        // perform the manual truncation

        Tensor newS = Tensor({trunc_dim}, S.dtype(), S.device());
        memcpy((cytnx_double *)newS._impl->storage()._impl->data(),
               (cytnx_double *)S._impl->storage()._impl->data(), trunc_dim * sizeof(cytnx_double));
        if (is_U) {
          Tensor newU = Tensor({U.shape()[0], trunc_dim}, U.dtype(), U.device());

          int src = 0;
          int dest = 0;
          // copy with strides.
          for (int i = 0; i < U.shape()[0]; i++) {
            memcpy((cytnx_float *)newU._impl->storage()._impl->data() + src,
                   (cytnx_float *)U._impl->storage()._impl->data() + dest,
                   trunc_dim * sizeof(cytnx_float));
            src += trunc_dim;
            dest += U.shape()[1];
          }
          U = newU;
        }
        if (is_vT) {
          Tensor newvT = Tensor({trunc_dim, vT.shape()[1]}, vT.dtype(), vT.device());
          // simply copy a new one dropping the tail.
          memcpy((cytnx_float *)newvT._impl->storage()._impl->data(),
                 (cytnx_float *)vT._impl->storage()._impl->data(),
                 vT.shape()[1] * trunc_dim * sizeof(cytnx_float));
          vT = newvT;
        }
        if (return_err == 1) {
          Tensor newterr = Tensor({1}, S.dtype(), S.device());
          ((cytnx_double *)newterr._impl->storage()._impl->data())[0] =
            ((cytnx_double *)S._impl->storage()._impl->data())[trunc_dim];
          terr = newterr;
        } else if (return_err) {
          cytnx_uint64 discared_dim = S.shape()[0] - trunc_dim;
          Tensor newterr = Tensor({discared_dim}, S.dtype(), S.device());
          memcpy((cytnx_double *)newterr._impl->storage()._impl->data(),
                 (cytnx_double *)S._impl->storage()._impl->data() + trunc_dim,
                 discared_dim * sizeof(cytnx_double));
          terr = newterr;
        }
        S = newS;
      }
    }

  }  // namespace linalg_internal
}  // namespace cytnx
