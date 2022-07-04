#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include <iostream>
#include "Tensor.hpp"
#include "Generator.hpp"

namespace cytnx {
  namespace linalg {
    std::vector<Tensor> Lstsq(const Tensor &A, const Tensor &b, const float &rcond) {
      cytnx_error_msg(A.shape().size() != 2 || b.shape().size() != 2,
                      "[Lsq] error, Lstsq can only operate on rank-2 Tensor.%s", "\n");

      cytnx_error_msg(A.device() != b.device(),
                      "[Lsq] error, A and b should be on the same device!%s", "\n");

      cytnx_uint64 m = A.shape()[0];
      cytnx_uint64 n = A.shape()[1];
      cytnx_uint64 nrhs = b.shape()[1];

      Tensor Ain;
      Tensor bin;
      if (A.is_contiguous())
        Ain = A.clone();
      else
        Ain = A.contiguous();

      if (b.is_contiguous())
        bin = b.clone();
      else
        bin = b.contiguous();

      int type_ = A.dtype() < b.dtype() ? A.dtype() : b.dtype();

      if (type_ > Type.Float) {
        type_ = Type.Double;  // if the strongest type is < int, then convert to double
      }

      Ain = Ain.astype(type_);
      bin = bin.astype(type_);

      if (m < n) {
        Storage bstor = bin.storage();
        bstor.resize(n * nrhs);
        bin = Tensor::from_storage(bstor).reshape(n, nrhs);
      }

      std::vector<Tensor> out;

      Tensor s =
        zeros(m < n ? m : n, Ain.dtype() <= 2 ? Ain.dtype() + 2 : Ain.dtype(), Ain.device());
      Tensor r = zeros(1, Type.Int64, Ain.device());

      if (A.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Lstsq_ii[Ain.dtype()](
          Ain._impl->storage()._impl, bin._impl->storage()._impl, s._impl->storage()._impl,
          r._impl->storage()._impl, m, n, nrhs, rcond);

        Tensor sol = bin(Accessor::range(0, n, 1), ":");
        sol.reshape_(n, nrhs);
        out.push_back(sol);

        Tensor res = zeros({1}, bin.dtype(), bin.device());
        if (m > n && r.item<cytnx_int64>() >= n) {
          Tensor res_ = bin(Accessor::range(n, m, 1), ":");
          res_.Pow_(2);
          Tensor ones_ = ones({1, static_cast<cytnx_uint64>(m - n)});
          res = linalg::Dot(ones_, res_);
        }
        out.push_back(res);
        out.push_back(r);
        out.push_back(s);
        return out;

      } else {
#ifdef UNI_GPU
        cytnx_error_msg(
          true, "[ERROR] currently Lstsq for non-symmetric matrix is not supported.%s", "\n");
        return std::vector<Tensor>();
#else
        cytnx_error_msg(true, "[Lsq] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return std::vector<Tensor>();
#endif
      }
    }
  }  // namespace linalg
}  // namespace cytnx
