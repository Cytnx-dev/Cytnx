#ifndef __H_vec_intersect_
#define __H_vec_intersect_

#include <vector>
#include "Type.hpp"
namespace cytnx{

    template<class T>
    std::vector<T> vec_intersect(const std::vector<T>& inL, const std::vector<T> &inR);
   
    template<class T>
    void vec_intersect_(std::vector<T> &out, const std::vector<T>& inL, const std::vector<T> &inR, std::vector<cytnx_uint64> &indices_v1, std::vector<cytnx_uint64> &indices_v2);


    template<class T>
    void vec_intersect_(std::vector<T> &out, const std::vector<T>& inL, const std::vector<T> &inR);

}
#endif
