#include "algo.hpp"
#include "Accessor.hpp"
#include "Generator.hpp"

#ifdef BACKEND_TORCH
#else

  #include "backend/Storage.hpp"
  #include "backend/Scalar.hpp"
namespace cytnx {
  namespace algo {
    typedef Accessor ac;

    Tensor _fx_Matric_combine(std::vector<Tensor> &Tlist, const cytnx_uint64 &Row_Nblk,
                              const cytnx_uint64 &Col_Nblk) {
      std::vector<Tensor> rows;
      rows.reserve(Row_Nblk);
      for (int r = 0; r < Row_Nblk; r++) {
        std::vector<Tensor> tmp(Tlist.begin() + r * Col_Nblk, Tlist.begin() + (r + 1) * Col_Nblk);
        rows.push_back(algo::Hstack(tmp));
      }
      return algo::Vstack(rows);
    }

    void _fx_Matric_split(std::vector<Tensor> &out, const Tensor &Tin,
                          const std::vector<cytnx_uint64> &rdims,
                          const std::vector<cytnx_uint64> &cdims) {
      // out.reserve(rdims.size()*cdims.size());
      out.clear();
      std::vector<Tensor> tmp;
      algo::Vsplit_(tmp, Tin, rdims);

      for (int i = 0; i < tmp.size(); i++) {
        auto tmpv = algo::Hsplit(tmp[i], cdims);
        out.insert(out.end(), tmpv.begin(), tmpv.end());
      }
    }

  }  // namespace algo
}  // namespace cytnx

#endif
