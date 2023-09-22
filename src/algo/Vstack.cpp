#include "algo.hpp"

#include "Accessor.hpp"
#include "Generator.hpp"

#ifdef BACKEND_TORCH
#else

  #include "backend/algo_internal_interface.hpp"
  #include "backend/Storage.hpp"
  #include "backend/Scalar.hpp"
namespace cytnx {
  namespace algo {
    typedef Accessor ac;

    Tensor Vstack(const std::vector<Tensor> &In_tensors) {
      Tensor out;

      std::vector<Tensor> _Ins;

      // check:
      // 1. Type matching
      // 2. all tensors must be rank-2!
      // 3. device matching
      // 4. dimension matching
      cytnx_error_msg(In_tensors.size() == 0, "[ERROR][Vstack] cannot have empty input list!%s",
                      "\n");
      cytnx_error_msg(In_tensors[0].shape().size() != 2,
                      "[ERROR][Vstack] elem: [0], Vstack can only work for rank-2 tensors.%s",
                      "\n");

      unsigned int dtype_id = In_tensors[0].dtype();
      int device_id = In_tensors[0].device();
      cytnx_uint64 Dshare = In_tensors[0].shape()[1];
      std::vector<cytnx_uint64> Ds(In_tensors.size());
      Ds[0] = In_tensors[0].storage().size();
      cytnx_uint64 Dcomb = In_tensors[0].shape()[0];

      bool need_convert = false;

      // checking:
      for (int i = 1; i < In_tensors.size(); i++) {
        if (In_tensors[i].dtype() != dtype_id) {
          need_convert = true;
          if (In_tensors[i].dtype() < dtype_id) {
            dtype_id = In_tensors[i].dtype();
          }
        }
        cytnx_error_msg(
          In_tensors[i].device() != device_id,
          "[ERROR][Vstack] elem: [%d], Vstack need all the tensors on the same device!\n", i);
        cytnx_error_msg(In_tensors[i].shape().size() != 2,
                        "[ERROR][Vstack] elem: [%d], Vstack can only work for rank-2 tensors.\n",
                        i);
        cytnx_error_msg(In_tensors[i].shape()[1] != Dshare,
                        "[ERROR][Vstack] elem: [%d], dimension not match. should be %d but is %d\n",
                        i, Dshare, In_tensors[i].shape()[1]);
        Ds[i] = In_tensors[i].storage().size();
        Dcomb += In_tensors[i].shape()[0];
      }
      cytnx_error_msg(dtype_id == Type.Bool,
                      "[ERROR][Vstack] currently does not support Bool type!%s", "\n");

      // conversion type:
      if (need_convert) {
        _Ins.resize(In_tensors.size());
        for (int i = 0; i < In_tensors.size(); i++) {
          if (In_tensors[i].dtype() != dtype_id) {
            _Ins[i] = In_tensors[i].astype(dtype_id).contiguous();
          } else {
            _Ins[i] = In_tensors[i].contiguous();
          }
        }

      } else {
        _Ins = In_tensors;
        for (int i = 0; i < In_tensors.size(); i++) {
          _Ins[i].contiguous_();
        }
      }

      // allocate out!
      out = zeros({Dcomb, Dshare}, dtype_id, device_id);

      std::vector<void *> rawPtr(In_tensors.size());
      for (int i = 0; i < _Ins.size(); i++) {
        rawPtr[i] = _Ins[i].storage().data();
      }

      if (device_id == Device.cpu) {
        algo_internal::vConcate_internal((char *)out.storage().data(), rawPtr, Ds,
                                         Type.typeSize(dtype_id));
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(device_id));
        algo_internal::cuvConcate_internal((char *)out.storage().data(), rawPtr, Ds,
                                           Type.typeSize(dtype_id));
  #else
        cytnx_error_msg(
          true, "[ERROR][VStack_] input is on GPU but current cytnx is compiled without GPU.%s",
          "\n");
  #endif
      }

      return out;
    }
  }  // namespace algo
}  // namespace cytnx

#endif
