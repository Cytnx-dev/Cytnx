#include "linalg.hpp"

#include <iostream>
#include "Tensor.hpp"
#include "Generator.hpp"

#ifdef BACKEND_TORCH
#else
  #include "../backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {

    void Gemm_(const Scalar &a, const Tensor &x, const Tensor &y, const Scalar &b, Tensor &c) {
      // C = a*x*y + b*C
      cytnx_error_msg(x.shape().size() != 2,
                      "[Gemm_] error, tensor x , Gemm can only operate on rank-2 Tensor.%s", "\n");
      cytnx_error_msg(y.shape().size() != 2,
                      "[Gemm_] error, tensor y , Gemm can only operate on rank-2 Tensor.%s", "\n");

      cytnx_error_msg(c.shape().size() != 2,
                      "[Gemm_] error, tensor c , Gemm can only operate on rank-2 Tensor.%s", "\n");

      cytnx_error_msg(x.device() != y.device(),
                      "[Gemm_] error tensors should all on same device. x.dev!=y.dev%s", "\n");
      cytnx_error_msg(y.device() != c.device(),
                      "[Gemm_] error tensors should all on same device. y.dev!=c.dev%s", "\n");

      // check dimension match
      cytnx_error_msg(x.shape()[1] != y.shape()[0], "[Gemm_] error, x,y dimension not match.%s",
                      "\n");
      cytnx_error_msg(x.shape()[0] != c.shape()[0], "[Gemm_] error, x,c dimension not match.%s",
                      "\n");
      cytnx_error_msg(y.shape()[1] != c.shape()[1], "[Gemm_] error, y,c dimension not match.%s",
                      "\n");

      // checking the largest dtype!
      int fin_dtype = x.dtype();
      if (y.dtype() < fin_dtype) fin_dtype = y.dtype();
      if (a.dtype() < fin_dtype) fin_dtype = a.dtype();
      if (b.dtype() < fin_dtype) fin_dtype = b.dtype();
      if (c.dtype() < fin_dtype) fin_dtype = c.dtype();

      // check Void type
      cytnx_error_msg(x.dtype() == Type.Void,
                      "[Gemm_] error tensor x with Type.Void cannot perform arithmetic.%s", "\n");
      cytnx_error_msg(y.dtype() == Type.Void,
                      "[Gemm_] error tensor y with Type.Void cannot perform arithmetic.%s", "\n");
      cytnx_error_msg(c.dtype() == Type.Void,
                      "[Gemm_] error tensor c with Type.Void cannot perform arithmetic.%s", "\n");
      cytnx_error_msg(a.dtype() == Type.Void,
                      "[Gemm_] error scalar a with Type.Void cannot perform arithmetic.%s", "\n");
      cytnx_error_msg(b.dtype() == Type.Void,
                      "[Gemm_] error scalar b with Type.Void cannot perform arithmetic.%s", "\n");

      // convert to double if dtype > 4
      if (fin_dtype > 4) {
        fin_dtype = Type.Double;
      }

      // convert dtype:
      Tensor px, py;
      if (x.dtype() > fin_dtype)
        px = x.astype(fin_dtype);
      else
        px = x;

      if (y.dtype() > fin_dtype)
        py = y.astype(fin_dtype);
      else
        py = y;

      Scalar pa, pb;
      if (a.dtype() > fin_dtype)
        pa = a.astype(fin_dtype);
      else
        pa = a;

      if (b.dtype() > fin_dtype)
        pb = b.astype(fin_dtype);
      else
        pb = b;

      // output change type!
      if (c.dtype() > fin_dtype) c = c.astype(fin_dtype);

      // contiguous?
      if (!px.is_contiguous()) px = px.contiguous();
      if (!py.is_contiguous()) py = py.contiguous();
      if (!c.is_contiguous()) c = c.contiguous();

      if (x.device() == Device.cpu) {
        linalg_internal::lii.Gemm_ii[fin_dtype](c._impl->storage()._impl, px._impl->storage()._impl,
                                                py._impl->storage()._impl, px.shape()[0],
                                                px.shape()[1], py.shape()[1], pa, pb);
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(x.device()));
        linalg_internal::lii.cuGemm_ii[fin_dtype](
          c._impl->storage()._impl, px._impl->storage()._impl, py._impl->storage()._impl,
          px.shape()[0], px.shape()[1], py.shape()[1], pa, pb);
          // cytnx_error_msg(true, "[Gemm_] fatal error,%s", "Not yet implemented.\n");
  #else
        cytnx_error_msg(true, "[Gemm_] fatal error,%s",
                        "try to use GPU but not compiled with GPU support.\n");
  #endif
      }
    }

    Tensor Gemm(const Scalar &a, const Tensor &x, const Tensor &y) {
      // ax*y -> out

      // std::cout << "matmul" << std::endl;
      // std::cout << Tl << Tr << std::endl;

      cytnx_error_msg(x.shape().size() != 2,
                      "[Gemm_] error, tensor x , Gemm can only operate on rank-2 Tensor.%s", "\n");
      cytnx_error_msg(y.shape().size() != 2,
                      "[Gemm_] error, tensor y , Gemm can only operate on rank-2 Tensor.%s", "\n");

      cytnx_error_msg(x.device() != y.device(),
                      "[Gemm_] error tensors should all on same device. x.dev!=y.dev%s", "\n");

      // check dimension match
      cytnx_error_msg(x.shape()[1] != y.shape()[0], "[Gemm_] error, x,y dimension not match.%s",
                      "\n");

      // checking the largest dtype!
      int fin_dtype = x.dtype();
      if (y.dtype() < fin_dtype) fin_dtype = y.dtype();
      if (a.dtype() < fin_dtype) fin_dtype = a.dtype();

      // convert dtype:
      Tensor px, py;
      if (x.dtype() > fin_dtype)
        px = x.astype(fin_dtype);
      else
        px = x;

      if (y.dtype() > fin_dtype)
        py = y.astype(fin_dtype);
      else
        py = y;

      Scalar pa;
      if (a.dtype() > fin_dtype)
        pa = a.astype(fin_dtype);
      else
        pa = a;

      // contiguous?
      if (!px.is_contiguous()) px = px.contiguous();
      if (!py.is_contiguous()) py = py.contiguous();

      Tensor out = zeros({px.shape()[0], py.shape()[1]}, fin_dtype, x.device());

      Scalar pb(1, fin_dtype);

      if (x.device() == Device.cpu) {
        linalg_internal::lii.Gemm_ii[fin_dtype](
          out._impl->storage()._impl, px._impl->storage()._impl, py._impl->storage()._impl,
          px.shape()[0], px.shape()[1], py.shape()[1], pa, pb);
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(x.device()));
        linalg_internal::lii.cuGemm_ii[fin_dtype](
          out._impl->storage()._impl, px._impl->storage()._impl, py._impl->storage()._impl,
          px.shape()[0], px.shape()[1], py.shape()[1], pa, pb);
          // cytnx_error_msg(true, "[Gemm_] fatal error,%s", "Not yet implemented.\n");
  #else
        cytnx_error_msg(true, "[Gemm_] fatal error,%s",
                        "try to use GPU but not compiled with GPU support.\n");
  #endif
      }

      return out;
    }

  }  // namespace linalg
}  // namespace cytnx

#endif
