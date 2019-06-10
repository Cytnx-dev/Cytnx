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

    template<class T>
    std::vector<T> vec_clone(const std::vector<T>& in_vec, const cytnx_uint64 &Nelem){
        cytnx_error_msg(Nelem > in_vec.size(),"[ERROR] Nelem cannot exceed the no. of elements in the in_vec%s","\n");
        std::vector<T> out(Nelem);
        for(cytnx_uint64 i=0;i<Nelem;i++){
            out[i] = in_vec[i].clone();
        }
        return out;
    }

    template<class T>
    std::vector<T> vec_clone(const std::vector<T>& in_vec, const std::vector<cytnx_uint64> &locators ){
        std::vector<T> out(locators.size());
        for(cytnx_uint64 i=0;i<locators.size();i++){
            cytnx_error_msg(locators[i] >= in_vec.size(),"[ERROR] the index [%d] in locators exceed the bbound.\n",locators[i]);
            out[i] = in_vec[locators[i]].clone();
        }
        return out;
    }


    template std::vector<Bond> vec_clone<Bond>(const std::vector<Bond>&);
    template std::vector<Symmetry> vec_clone<Symmetry>(const std::vector<Symmetry>&);
    template std::vector<Tensor> vec_clone<Tensor>(const std::vector<Tensor>&);
    template std::vector<Storage> vec_clone<Storage>(const std::vector<Storage>&);

    template std::vector<Bond> vec_clone<Bond>(const std::vector<Bond>&, const cytnx_uint64 &);
    template std::vector<Symmetry> vec_clone<Symmetry>(const std::vector<Symmetry>&, const cytnx_uint64 &);
    template std::vector<Tensor> vec_clone<Tensor>(const std::vector<Tensor>&, const cytnx_uint64 &);
    template std::vector<Storage> vec_clone<Storage>(const std::vector<Storage>&, const cytnx_uint64 &);


    template std::vector<Bond> vec_clone<Bond>(const std::vector<Bond>&, const std::vector<cytnx_uint64> &);
    template std::vector<Symmetry> vec_clone<Symmetry>(const std::vector<Symmetry>&, const std::vector<cytnx_uint64> &);
    template std::vector<Tensor> vec_clone<Tensor>(const std::vector<Tensor>&, const std::vector<cytnx_uint64> &);
    template std::vector<Storage> vec_clone<Storage>(const std::vector<Storage>&, const std::vector<cytnx_uint64> &);



}
