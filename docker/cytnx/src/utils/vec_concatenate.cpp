#include "utils/vec_concatenate.hpp"
#include "Bond.hpp"
#include <algorithm>
#include <vector>
#include <cstring>
namespace cytnx{

    template<class T>
    std::vector<T> vec_concatenate(const std::vector<T>& inL, const std::vector<T> &inR){
        std::vector<T> out(inL.size()+inR.size());
        memcpy(&out[0],&inL[0],sizeof(T)*inL.size());
        memcpy(&out[inL.size()], &inR[0],sizeof(T)*inR.size());
        return out;
    }

    template<class T>
    void vec_concatenate_(std::vector<T> &out, const std::vector<T> &inL, const std::vector<T> &inR){
        out.resize(inL.size()+inR.size());
        memcpy(&out[0],&inL[0],sizeof(T)*inL.size());
        memcpy(&out[inL.size()], &inR[0],sizeof(T)*inR.size());
    }


    template std::vector<cytnx_complex128> vec_concatenate(const std::vector<cytnx_complex128> &,const std::vector<cytnx_complex128> &);
    template std::vector<cytnx_complex64> vec_concatenate(const std::vector<cytnx_complex64> &,const std::vector<cytnx_complex64> &);
    template std::vector<cytnx_double> vec_concatenate(const std::vector<cytnx_double> &,const std::vector<cytnx_double> &);
    template std::vector<cytnx_float> vec_concatenate(const std::vector<cytnx_float> &,const std::vector<cytnx_float> &);
    template std::vector<cytnx_int64> vec_concatenate(const std::vector<cytnx_int64> &,const std::vector<cytnx_int64> &);
    template std::vector<cytnx_uint64> vec_concatenate(const std::vector<cytnx_uint64> &,const std::vector<cytnx_uint64> &);
    template std::vector<cytnx_int32> vec_concatenate(const std::vector<cytnx_int32> &,const std::vector<cytnx_int32> &);
    template std::vector<cytnx_uint32> vec_concatenate(const std::vector<cytnx_uint32> &,const std::vector<cytnx_uint32> &);

    template void vec_concatenate_(std::vector<cytnx_complex128> &out, const std::vector<cytnx_complex128> &,const std::vector<cytnx_complex128> &);
    template void vec_concatenate_(std::vector<cytnx_complex64> &out,const std::vector<cytnx_complex64> &,const std::vector<cytnx_complex64> &);
    template void vec_concatenate_(std::vector<cytnx_double> &out,const std::vector<cytnx_double> &,const std::vector<cytnx_double> &);
    template void vec_concatenate_(std::vector<cytnx_float> &out,const std::vector<cytnx_float> &,const std::vector<cytnx_float> &);
    template void vec_concatenate_(std::vector<cytnx_int64> &out,const std::vector<cytnx_int64> &,const std::vector<cytnx_int64> &);
    template void vec_concatenate_(std::vector<cytnx_uint64> &out,const std::vector<cytnx_uint64> &,const std::vector<cytnx_uint64> &);
    template void vec_concatenate_(std::vector<cytnx_int32> &out,const std::vector<cytnx_int32> &,const std::vector<cytnx_int32> &);
    template void vec_concatenate_(std::vector<cytnx_uint32> &out,const std::vector<cytnx_uint32> &,const std::vector<cytnx_uint32> &);

}
