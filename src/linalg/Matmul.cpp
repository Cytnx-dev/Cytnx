#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include <iostream>
#include "Tensor.hpp"
#include "lapack_wrapper.hpp"

namespace cytnx {
  namespace linalg {
    Tensor Matmul(const Tensor &Tl, const Tensor &Tr) {
      // std::cout << "matmul" << std::endl;
      // std::cout << Tl << Tr << std::endl;

      cytnx_error_msg(Tl.shape().size() != 2,
                      "[Matmul] error, tensor Tl ,Matmul can only operate on rank-2 Tensor.%s",
                      "\n");
      cytnx_error_msg(Tr.shape().size() != 2,
                      "[Matmul] error, tensor Tr ,Matmul can only operate on rank-2 Tensor.%s",
                      "\n");

      // cytnx_error_msg(!Tl.is_contiguous(), "[Matmul] error tensor Tl must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      // cytnx_error_msg(!Tr.is_contiguous(), "[Matmul] error tensor Tr must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      // check device:
      cytnx_error_msg(Tl.device() != Tr.device(),
                      "[Matmul] error two tensor should be on same device.%s", "\n");

      // check dimension match
      cytnx_error_msg(Tl.shape()[1] != Tr.shape()[0], "[Matmul] error, dimension not match.%s",
                      "\n");

      // check type:
      Tensor _tl = Tl.contiguous(), _tr = Tr.contiguous();
      Tensor out;
      if (Tl.dtype() != Tr.dtype()) {
        // do conversion:
        if (Tl.dtype() < Tr.dtype()) {
          _tr = _tr.astype(Tl.dtype());
          out.Init({Tl.shape()[0], Tr.shape()[1]}, Tl.dtype(), Tl.device());
        } else {
          _tl = _tl.astype(Tr.dtype());
          out.Init({Tl.shape()[0], Tr.shape()[1]}, Tr.dtype(), Tr.device());
        }
      } else {
        out.Init({Tl.shape()[0], Tr.shape()[1]}, Tr.dtype(), Tr.device(), false);
      }
      // out.storage().set_zeros();

      if (Tl.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Matmul_ii[_tl.dtype()](
          out._impl->storage()._impl, _tl._impl->storage()._impl, _tr._impl->storage()._impl,
          _tl.shape()[0], _tl.shape()[1], _tr.shape()[1]);

        return out;

      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tl.device()));
        cytnx::linalg_internal::lii.cuMatmul_ii[_tl.dtype()](
          out._impl->storage()._impl, _tl._impl->storage()._impl, _tr._impl->storage()._impl,
          _tl.shape()[0], _tl.shape()[1], _tr.shape()[1]);

        // std::cout << "GPU Matmul OUT" << std::endl;
        // std::cout << out <<std::endl;
        // std::cout << "xxxxxxxxxxxxxx\n";
        return out;
#else
        cytnx_error_msg(true, "[Matmul] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return Tensor();
#endif
      }
    }

    void d_Matmul(Tensor &Tl, Tensor &Tr, Tensor &out,
                const cytnx_double alpha, const cytnx_double beta,
                const bool check){
      cytnx_error_msg(check && Tl.shape().size() != 2,
                      "[Matmul] error, tensor Tl ,Matmul can only operate on rank-2 Tensor.%s",
                      "\n");
      cytnx_error_msg(check && Tr.shape().size() != 2,
                      "[Matmul] error, tensor Tr ,Matmul can only operate on rank-2 Tensor.%s",
                      "\n");
      cytnx_error_msg(check && out.shape().size() != 2,
                      "[Matmul] error, tensor out ,Matmul can only operate on rank-2 Tensor.%s",
                      "\n");
      cytnx_error_msg(check && !Tl.is_contiguous(),
                      "[Matmul] error tensor Tl must be contiguous. Call contiguous_() or contiguous() first!%s",
                      "\n");
      cytnx_error_msg(check && !Tr.is_contiguous(),
                      "[Matmul] error tensor Tr must be contiguous. Call contiguous_() or contiguous() first!%s",
                      "\n");
      cytnx_error_msg(check && !out.is_contiguous(),
                      "[Matmul] error tensor out must be contiguous. Call contiguous_() or contiguous() first!%s",
                      "\n");

      // check device:
      cytnx_error_msg(check && Tl.device() != Tr.device(),
                      "[Matmul] error two tensor should be on same device.%s", "\n");
      cytnx_error_msg(check && Tl.device() != out.device(),
                      "[Matmul] error two tensor should be on same device.%s", "\n");

      // check dimension match
      cytnx_error_msg(check && Tl.shape()[1] != Tr.shape()[0], "[Matmul] error, dimension not match.%s",
                      "\n");

      if (Tl.device() == Device.cpu) {
        auto out_ = out._impl->storage()._impl;
        auto Tl_ = Tl._impl->storage()._impl;
        auto Tr_ = Tr._impl->storage()._impl;
        cytnx_double *_out = (cytnx_double *)out_->Mem;
        cytnx_double *_inl = (cytnx_double *)Tl_->Mem;
        cytnx_double *_inr = (cytnx_double *)Tr_->Mem;

        // cytnx_double alpha = alpha, beta = beta;
        blas_int blsMl = Tl.shape()[0], blsNr = Tr.shape()[1], blsComm = Tl.shape()[1];
        dgemm((char *)"N", (char *)"N", &blsNr, &blsMl, &blsComm, &alpha, _inr, &blsNr, _inl,
              &blsComm, &beta, _out, &blsNr);
        return;
      } 
      else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tl.device()));
        cytnx::linalg_internal::lii.cuMatmul_ii[_tl.dtype()](
          out._impl->storage()._impl, Tl._impl->storage()._impl, Tl._impl->storage()._impl,
          Tl.shape()[0], Tl.shape()[1], Tl.shape()[1]);

        // std::cout << "GPU Matmul OUT" << std::endl;
        // std::cout << out <<std::endl;
        // std::cout << "xxxxxxxxxxxxxx\n";
        return;
#else
        cytnx_error_msg(true, "[Matmul] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return;
#endif
      }
    }

    void cd_Matmul(Tensor &Tl, Tensor &Tr, Tensor &out,
                const cytnx_complex128 alpha, const cytnx_complex128 beta,
                const bool check){
      cytnx_error_msg(check && Tl.shape().size() != 2,
                      "[Matmul] error, tensor Tl ,Matmul can only operate on rank-2 Tensor.%s",
                      "\n");
      cytnx_error_msg(check && Tr.shape().size() != 2,
                      "[Matmul] error, tensor Tr ,Matmul can only operate on rank-2 Tensor.%s",
                      "\n");
      cytnx_error_msg(check && !Tl.is_contiguous(),
                      "[Matmul] error tensor Tl must be contiguous. Call contiguous_() or contiguous() first!%s",
                      "\n");
      cytnx_error_msg(check && !Tr.is_contiguous(),
                      "[Matmul] error tensor Tr must be contiguous. Call contiguous_() or contiguous() first!%s",
                      "\n");

      // check device:
      cytnx_error_msg(check && Tl.device() != Tr.device(),
                      "[Matmul] error two tensor should be on same device.%s", "\n");

      // check dimension match
      cytnx_error_msg(check && Tl.shape()[1] != Tr.shape()[0], "[Matmul] error, dimension not match.%s",
                      "\n");

      if (Tl.device() == Device.cpu) {
        cytnx_complex128 *_out = (cytnx_complex128 *)out._impl->storage()._impl->Mem;
        cytnx_complex128 *_inl = (cytnx_complex128 *)Tl._impl->storage()._impl->Mem;
        cytnx_complex128 *_inr = (cytnx_complex128 *)Tr._impl->storage()._impl->Mem;

        // cytnx_double alpha = alpha, beta = beta;
        blas_int blsMl = Tl.shape()[0], blsNr = Tr.shape()[1], blsComm = Tl.shape()[1];
        zgemm((char *)"N", (char *)"N", &blsNr, &blsMl, &blsComm, &alpha, _inr, &blsNr, _inl,
              &blsComm, &beta, _out, &blsNr);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tl.device()));
        cytnx::linalg_internal::lii.cuMatmul_ii[_tl.dtype()](
          out._impl->storage()._impl, Tl._impl->storage()._impl, Tl._impl->storage()._impl,
          Tl.shape()[0], Tl.shape()[1], Tl.shape()[1]);

        // std::cout << "GPU Matmul OUT" << std::endl;
        // std::cout << out <<std::endl;
        // std::cout << "xxxxxxxxxxxxxx\n";
        return;
#else
        cytnx_error_msg(true, "[Matmul] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return;
#endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx
