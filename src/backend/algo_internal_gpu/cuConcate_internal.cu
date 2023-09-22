#include "cuConcate_internal.hpp"

namespace cytnx {

  namespace algo_internal {

    void cuvConcate_internal(char *out_ptr, std::vector<void *> &ins,
                             const std::vector<cytnx_uint64> &lens, const cytnx_uint64 &ElemSize) {
      // require:
      // 1. Data type of out, all [ins] to be the same
      // 2. out is properly allocated!
      // 4. checking bool type!!

      // cytnx_uint64 ElemSize = Type.typeSize(out->dtype);
      cytnx_uint64 offs = 0;
      // char *out_ptr = (char*)out->Mem;

      for (cytnx_int32 i = 0; i < ins.size(); i++) {
        checkCudaErrors(
          cudaMemcpy(out_ptr + offs, ins[i], ElemSize * lens[i], cudaMemcpyDeviceToDevice));
        offs += ElemSize * lens[i];
        // std::cout << ElemSize*lens[i] << std::endl;
      }
    }

    void cuhConcate_internal(char *out_ptr, std::vector<char *> &ins,
                             const std::vector<cytnx_uint64> &lens, const cytnx_uint64 &Dshare,
                             const cytnx_uint64 &Dtot, const cytnx_uint64 &ElemSize) {
      // require:
      // 1. Data type of out, all [ins] to be the same
      // 2. out is properly allocated!
      // 4. checking bool type!!

      cytnx_uint64 off = 0;
      std::vector<cytnx_uint64> offs(lens.size());
      for (int i = 0; i < offs.size(); i++) {
        offs[i] = off;
        off += lens[i];
      }

      // iterate each small chunk:
      for (cytnx_int64 t = 0; t < lens.size(); t++) {
        // copy segment for each row!
        for (cytnx_int64 r = 0; r < Dshare; r++) {
          checkCudaErrors(cudaMemcpy(out_ptr + (Dtot * r + offs[t]) * ElemSize,
                                     ins[t] + (r * lens[t]) * ElemSize, ElemSize * lens[t],
                                     cudaMemcpyDeviceToDevice));
        }  // r

      }  // t
    }

  }  // namespace algo_internal

}  // namespace cytnx
