#include "utils/vec_clone.hpp"
#include "Tensor.hpp"
#include "Storage.hpp"
#include "Symmetry.hpp"
#include "Bond.hpp"
namespace cytnx{

    template<class T>
    std::vector<T> vec_clone(const std::vector<T>& in_vec){
        std::vector<T> out(in_vec.size());
        for(cytnx_uint64 i=0;i<in_vec.size();i++){
            out[i] = in_vec[i].clone();
        }
        return out;
    }

    template std::vector<Bond> vec_clone<Bond>(const std::vector<Bond>&);
    template std::vector<Symmetry> vec_clone<Symmetry>(const std::vector<Symmetry>&);
    template std::vector<Tensor> vec_clone<Tensor>(const std::vector<Tensor>&);
    template std::vector<Storage> vec_clone<Storage>(const std::vector<Storage>&);

}
