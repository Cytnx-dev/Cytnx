#include "linalg.hpp"
#include "utils/utils.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "../backend/linalg_internal_interface.hpp"

namespace cytnx {

  namespace linalg {

    void _Tensordot_generic(Tensor &out, const Tensor &Tl, const Tensor &Tr,
                            const std::vector<cytnx_uint64> &idxl,
                            const std::vector<cytnx_uint64> &idxr, const bool &cacheL,
                            const bool &cacheR) {
      std::vector<cytnx_uint64> mapperL, mapperR;
      std::vector<cytnx_uint64> non_contract_l = vec_erase(vec_range(Tl.shape().size()), idxl);
      std::vector<cytnx_uint64> non_contract_r = vec_erase(vec_range(Tr.shape().size()), idxr);

      // calculate permute
      vec_concatenate_(mapperL, non_contract_l, idxl);
      vec_concatenate_(mapperR, idxr, non_contract_r);

      // checking + calculate comm_dim:
      cytnx_int64 comm_dim = 1;
      for (cytnx_uint64 i = 0; i < idxl.size(); i++) {
        cytnx_error_msg(Tl.shape()[idxl[i]] != Tr.shape()[idxr[i]],
                        "the index L=%d and R=%d have different dimension!\n", idxl[i], idxr[i]);
        comm_dim *= Tl.shape()[idxl[i]];
      }

      // calculate output shape:
      std::vector<cytnx_int64> new_shape(non_contract_l.size() + non_contract_r.size());
      for (cytnx_uint64 i = 0; i < non_contract_l.size(); i++)
        new_shape[i] = Tl.shape()[non_contract_l[i]];
      for (cytnx_uint64 i = 0; i < non_contract_r.size(); i++)
        new_shape[non_contract_l.size() + i] = Tr.shape()[non_contract_r[i]];

      if (new_shape.size() == 0) {
        new_shape.push_back(1);
      }

      Tensor tmpL = Tl;
      Tensor tmpR = Tr;

      std::vector<cytnx_uint64> inv_mapperL, inv_mapperR;
      std::vector<cytnx_uint64> oldshapeL, oldshapeR;

      if (cacheL) {
        // calculate reverse mapper:
        inv_mapperL.resize(mapperL.size());
        for (int i = 0; i < mapperL.size(); i++) {
          inv_mapperL[mapperL[i]] = i;
        }
        tmpL.permute_(mapperL);
        oldshapeL = tmpL.shape();
        tmpL.reshape_({-1, comm_dim});

      } else {
        tmpL = Tl.permute(mapperL).reshape({-1, comm_dim});
      }
      if (cacheR) {
        // calculate reverse mapper:
        inv_mapperR.resize(mapperR.size());
        for (int i = 0; i < mapperR.size(); i++) {
          inv_mapperR[mapperR[i]] = i;
        }
        tmpR.permute_(mapperR);
        oldshapeR = tmpR.shape();
        tmpR.reshape_({comm_dim, -1});

      } else {
        tmpR = Tr.permute(mapperR).reshape({comm_dim, -1});
      }

      // permute!
      // Tensor tmpL = Tl.permute(mapperL).reshape({-1,comm_dim});
      // Tensor tmpR = Tr.permute(mapperR).reshape({comm_dim,-1});

      out = Matmul(tmpL, tmpR);
      out.reshape_(new_shape);

      if (cacheL) {
        tmpL.reshape_(oldshapeL);
        tmpL.permute_(inv_mapperL);
      }
      if (cacheR) {
        tmpR.reshape_(oldshapeR);
        tmpR.permute_(inv_mapperR);
      }
    }

  #ifdef UNI_GPU
    void _Tensordot_cutn(Tensor &out, const Tensor &Tl, const Tensor &Tr,
                         const std::vector<cytnx_uint64> &idxl,
                         const std::vector<cytnx_uint64> &idxr, const bool &cacheL,
                         const bool &cacheR) {
      unsigned int t = Tl.dtype();
      if (t == Type.Uint64 || t == Type.Int64 || t == Type.Uint32 || t == Type.Int32 ||
          t == Type.Uint16 || t == Type.Int16 || t == Type.Bool) {
        cytnx_warning_msg(true, "Unsupported data type in cuTensor: %s, use default implementation",
                          Type.Typeinfos[Tl.dtype()].name);
        return _Tensordot_generic(out, Tl, Tr, idxl, idxr, cacheL, cacheR);
      }
      // This works:
      // if (t != Tr.dtype()) {
      //   cytnx_warning_msg(true, "[Tensordot] Tl.dtype != Tr.dtype, use default implementation%s",
      //                     "\n");
      //   return _Tensordot_generic(out, Tl, Tr, idxl, idxr, cacheL, cacheR);
      // }

      Tensor _tl = Tl.contiguous(), _tr = Tr.contiguous();
      if (Tl.dtype() != Tr.dtype()) {
        // do conversion:
        if (Tl.dtype() < Tr.dtype()) {
          _tr = _tr.astype(Tl.dtype());
        } else {
          _tl = _tl.astype(Tr.dtype());
        }
      }

      // checking + calculate comm_dim:
      for (cytnx_uint64 i = 0; i < idxl.size(); i++) {
        cytnx_error_msg(_tl.shape()[idxl[i]] != _tr.shape()[idxr[i]],
                        "the index L=%d and R=%d have different dimension!\n", idxl[i], idxr[i]);
      }

      // check device:
      cytnx_error_msg(_tl.device() != _tr.device(),
                      "[Tensordot] error two tensor should be on same device.%s", "\n");

      std::vector<cytnx_uint64> non_contract_l = vec_erase(vec_range(_tl.shape().size()), idxl);
      std::vector<cytnx_uint64> non_contract_r = vec_erase(vec_range(_tr.shape().size()), idxr);

      // calculate output shape:
      std::vector<cytnx_uint64> new_shape(non_contract_l.size() + non_contract_r.size());
      for (cytnx_uint64 i = 0; i < non_contract_l.size(); i++)
        new_shape[i] = _tl.shape()[non_contract_l[i]];
      for (cytnx_uint64 i = 0; i < non_contract_r.size(); i++)
        new_shape[non_contract_l.size() + i] = _tr.shape()[non_contract_r[i]];

      if (new_shape.size() == 0) {
        new_shape.push_back(1);
      }

      out.Init(new_shape, _tr.dtype(), _tr.device(), false);

      checkCudaErrors(cudaSetDevice(_tl.device()));
      cytnx::linalg_internal::lii.cuTensordot_ii[_tl.dtype()](out, _tl, _tr, idxl, idxr);
    }
  #endif

    Tensor Tensordot(const Tensor &Tl, const Tensor &Tr, const std::vector<cytnx_uint64> &idxl,
                     const std::vector<cytnx_uint64> &idxr, const bool &cacheL,
                     const bool &cacheR) {
      // checking:
      cytnx_error_msg(idxl.size() != idxr.size(),
                      "[ERROR] the number of index to trace must be consist across two tensors.%s",
                      "\n");
      cytnx_error_msg(
        idxl.size() == 0,
        "[ERROR] pass empty index list for trace. suggestion: call linalg::Otimes() instead?%s",
        "\n");
      cytnx_error_msg(Tl.device() != Tr.device(),
                      "[ERROR] two tensor for Tensordot cannot on different devices.%s", "\n");

      // check if two tensor has same data, to prevent conflict!
      if (cacheL && cacheR) {
        cytnx_error_msg(Tl.same_data(Tr),
                        "[ERROR] tensordot with both mv_elem options = True cannot have both two "
                        "input tensors to be the same.%s",
                        "\n");
      }

      Tensor out;

      if (Tl.device() == Device.cpu) {
        _Tensordot_generic(out, Tl, Tr, idxl, idxr, cacheL, cacheR);

      } else {
  #ifdef UNI_GPU
    #ifdef UNI_CUTENSOR
        _Tensordot_cutn(out, Tl, Tr, idxl, idxr, cacheL, cacheR);
    #else
        _Tensordot_generic(out, Tl, Tr, idxl, idxr, cacheL, cacheR);
    #endif
  #else
        cytnx_error_msg(true, "calling GPU version of Tensordot without CUDA support!%s", "\n");
  #endif
      }

      return out;
    }

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
