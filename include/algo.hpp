#ifndef _algo_H_
#define _algo_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include "Storage.hpp"
#include "Scalar.hpp"

namespace cytnx {
  namespace algo {

    // Sort:
    //=================================================
    /**
    @brief sort Tensor along last axis.
    @param Tin Tensor #1
    @return
        [Tensor]

    */
    Tensor Sort(const Tensor &Tin);

    Tensor Concatenate(Tensor T1, Tensor T2);

    Tensor Vstack(const std::vector<Tensor> &In_tensors);
    Tensor Hstack(const std::vector<Tensor> &In_tensors);


    void Vsplit_(std::vector<Tensor> &out, const Tensor &Tin, const std::vector<cytnx_uint64> &dims);
    void Hsplit_(std::vector<Tensor> &out, const Tensor &Tin, const std::vector<cytnx_uint64> &dims);
    std::vector<Tensor> Vsplit( const Tensor &Tin, const std::vector<cytnx_uint64> &dims);
    std::vector<Tensor> Hsplit( const Tensor &Tin, const std::vector<cytnx_uint64> &dims);
    
    




    ///@cond
    Tensor _fx_Matric_combine(std::vector<Tensor> &Tlist, const cytnx_uint64 &Row_Nblk, const cytnx_uint64 &Col_Nblk);
    void _fx_Matric_split(std::vector<Tensor> &out, const Tensor &Tin, const std::vector<cytnx_uint64> &rdims, const std::vector<cytnx_uint64> &cdims);
    ///@endcond




  }  // namespace algo
}  // namespace cytnx

#endif
