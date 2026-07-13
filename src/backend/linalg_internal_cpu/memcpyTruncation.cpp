#include "memcpyTruncation.hpp"

#include <cstring>
#include <type_traits>
#include <variant>

#include "Generator.hpp"

namespace cytnx {
  namespace linalg_internal {

    void memcpyTruncation(std::vector<Tensor> &tens, cytnx_uint64 keepdim, double err, bool is_U,
                          bool is_vT, unsigned int return_err, cytnx_uint64 mindim) {
      // at least one singular value is always kept
      if (mindim < 1) mindim = 1;
      const cytnx_uint64 nums = tens[0].storage().size();
      cytnx_uint64 trunc_dim = (nums < keepdim) ? nums : keepdim;

      if (nums == 0) {
        if (return_err == 1) {
          tens.push_back(zeros({}, tens[0].dtype(), tens[0].device()));
        } else if (return_err) {
          tens.push_back(zeros({0}, tens[0].dtype(), tens[0].device()));
        }
        return;
      }

      // error tensor; filled below only when return_err != 0 (see header for its exact contents).
      Tensor terr;

      // --- S (and terr): real, so we visit once on its native float/double type and copy
      //     directly. The complex/integer variant alternatives cannot occur for S. ---
      Tensor S = tens[0].contiguous();
      std::visit(
        [&](auto sptr) {
          using T = std::remove_pointer_t<decltype(sptr)>;
          if constexpr (std::is_floating_point_v<T>) {
            // determine the truncation dimension
            for (cytnx_int64 i = trunc_dim - 1; i >= 0; --i) {
              if (sptr[i] < err && trunc_dim - 1 >= mindim) {
                trunc_dim--;
              } else {
                break;
              }
            }
            if (trunc_dim == 0) trunc_dim = 1;
            if (trunc_dim == nums) {
              // nothing to truncate; there are no discarded values, so the truncation error is
              // zero.
              if (return_err == 1) {
                terr = zeros({}, tens[0].dtype(), tens[0].device());
              } else if (return_err) {
                terr = zeros({0}, tens[0].dtype(), tens[0].device());
              }
              return;  // escapes lambda
            }
            // truncated singular values
            Tensor newS({trunc_dim}, S.dtype(), S.device());
            std::memcpy(newS.ptr_as<T>(), sptr, trunc_dim * sizeof(T));

            if (return_err == 1) {
              terr = Tensor({}, S.dtype(), S.device());
              terr.ptr_as<T>()[0] = sptr[trunc_dim];
            } else if (return_err) {
              const cytnx_uint64 discarded_dim = nums - trunc_dim;
              terr = Tensor({discarded_dim}, S.dtype(), S.device());
              std::memcpy(terr.ptr_as<T>(), sptr + trunc_dim, discarded_dim * sizeof(T));
            }

            tens[0] = newS;
          } else {
            cytnx_error_msg(true,
                            "[memcpyTruncation] singular values S must be a real floating-point "
                            "type, got %s.\n",
                            Type.getname(tens[0].dtype()).c_str());
          }
        },
        S.ptr());

      // --- U / vT: located positionally in the packed vector ([S, U?, vT?]); the byte copy is
      // dispatched on the matrix dtype via std::visit. Only needed when truncation happened. ---
      if (trunc_dim != nums) {
        cytnx_uint64 t = 1;
        if (is_U) {
          Tensor src = tens[t].contiguous();
          cytnx_uint64 rows = src.shape()[0], cols = src.shape()[1];
          Tensor newU({rows, trunc_dim}, src.dtype(), src.device());
          std::visit(
            [&](auto sptr) {
              using T = std::remove_pointer_t<decltype(sptr)>;
              T *dst = newU.ptr_as<T>();
              // keep the first trunc_dim columns of each row; row-major, so copy row by row
              for (cytnx_uint64 i = 0; i < rows; ++i) {
                std::memcpy(dst + i * trunc_dim, sptr + i * cols, trunc_dim * sizeof(T));
              }
            },
            src.ptr());
          tens[t] = newU;
          t++;
        }
        if (is_vT) {
          Tensor src = tens[t].contiguous();
          cytnx_uint64 cols = src.shape()[1];
          Tensor newvT({trunc_dim, cols}, src.dtype(), src.device());
          std::visit(
            [&](auto sptr) {
              using T = std::remove_pointer_t<decltype(sptr)>;
              // keeping the first trunc_dim rows is a single contiguous prefix copy
              std::memcpy(newvT.ptr_as<T>(), sptr, trunc_dim * cols * sizeof(T));
            },
            src.ptr());
          tens[t] = newvT;
        }
      }

      if (return_err) tens.push_back(terr);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
