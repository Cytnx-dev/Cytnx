#include "Tensor.hpp"
#include "utils/utils_internal.hpp"
#include "linalg/linalg.hpp"

using namespace std;

namespace cytnx{        

    void Tensor_impl::Init(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype, int device){
        //check:
        cytnx_error_msg(dtype>=N_Type,"%s","[ERROR] invalid argument: dtype");
        cytnx_error_msg(shape.size()==0,"%s","[ERROR] invalid argument: shape. Must at least have one element.");
        cytnx_uint64 Nelem= 1;
        for(int i=0;i<shape.size();i++){
            cytnx_error_msg(shape[i]==0,"%s","[ERROR] shape cannot have 0 dimension in any rank.");
            Nelem *= shape[i]; 
        }
        this->_storage = __SII.USIInit[dtype]();
        this->_storage->Init(Nelem,device);
        this->_shape = shape;
        this->_mapper = utils_internal::range_cpu(shape.size());
        this->_invmapper = this->_mapper;
        this->_contiguous = true;

    }

    void Tensor_impl::permute_(const std::vector<cytnx_uint64> &rnks){
        //check::
        if(rnks.size()!=this->_shape.size()){
            cytnx_error_msg(true,"%s","reshape a tensor with a specify shape that does not match with the shape of the incident tensor.");
        }

        std::vector<cytnx_uint64> new_fwdmap(this->_shape.size());
        std::vector<cytnx_uint64> new_shape(this->_shape.size());
        std::vector<cytnx_uint64> new_idxmap(this->_shape.size());

        //for(int i=0;i<this->_shape.size();i++)
        //    std::cout << this->_mapper[i] << " " << this->_invmapper[i] << std::endl;                


        for(cytnx_uint32 i=0;i<rnks.size();i++){
            if(rnks[i] >= rnks.size()){
                cytnx_error_msg(1,"%s","reshape a tensor with invalid rank index.");
            }
            //std::cout << this->_mapper[rnks[i]] << " " << i << std::endl;
            new_idxmap[this->_mapper[rnks[i]]] = i;
            new_fwdmap[i] = this->_mapper[rnks[i]];
            new_shape[i] = this->_shape[rnks[i]];

        }

        this->_invmapper = new_idxmap;
        this->_shape = new_shape;
        this->_mapper = new_fwdmap;

        ///checking if permute back to contiguous:
        bool iconti=true;
        for(cytnx_uint32 i=0;i<rnks.size();i++){
            if(new_fwdmap[i]!=new_idxmap[i]){iconti = false; break;}
            if(new_fwdmap[i] != i){iconti=false; break;}
        }
        this->_contiguous= iconti;
    }            

    std::ostream& operator<<(std::ostream& os, Tensor &in){
        if(in.is_contiguous()) in._impl->_get_storage()->PrintElem_byShape(os,in.shape());
        else in._impl->_get_storage()->PrintElem_byShape(os,in.shape(),in._impl->_get_invmapper());
        return os;
    }       

    //===================================================================
    //wrapper


    // += 
    template<> Tensor& Tensor::operator+=<Tensor>(const Tensor &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_complex128>(const cytnx_complex128 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_complex64>(const cytnx_complex64 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_double>(const cytnx_double &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_float>(const cytnx_float &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_int64>(const cytnx_int64 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_uint64>(const cytnx_uint64 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_int32>(const cytnx_int32 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_uint32>(const cytnx_uint32 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    // -= 
    template<> Tensor& Tensor::operator-=<Tensor>(const Tensor &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_complex128>(const cytnx_complex128 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_complex64>(const cytnx_complex64 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_double>(const cytnx_double &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_float>(const cytnx_float &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_int64>(const cytnx_int64 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_uint64>(const cytnx_uint64 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_int32>(const cytnx_int32 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_uint32>(const cytnx_uint32 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    // *= 
    template<> Tensor& Tensor::operator*=<Tensor>(const Tensor &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_complex128>(const cytnx_complex128 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_complex64>(const cytnx_complex64 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_double>(const cytnx_double &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_float>(const cytnx_float &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_int64>(const cytnx_int64 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_uint64>(const cytnx_uint64 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_int32>(const cytnx_int32 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_uint32>(const cytnx_uint32 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    // /= 
    template<> Tensor& Tensor::operator/=<Tensor>(const Tensor &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_complex128>(const cytnx_complex128 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_complex64>(const cytnx_complex64 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_double>(const cytnx_double &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_float>(const cytnx_float &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_int64>(const cytnx_int64 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_uint64>(const cytnx_uint64 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_int32>(const cytnx_int32 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_uint32>(const cytnx_uint32 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }



    std::vector<Tensor> Tensor::Svd(const bool&is_U, const bool&is_vT){
        return linalg::Svd(*this, is_U, is_vT);
    }

    Tensor& Tensor::Inv_(){
        linalg::Inv_(*this);
        return *this;
    }
    Tensor Tensor::Inv(){
        return linalg::Inv(*this); 
    }
    
    Tensor& Tensor::Conj_(){
        linalg::Conj_(*this);
        return *this;
    }
    Tensor Tensor::Conj(){
        return linalg::Conj(*this); 
    }

    Tensor& Tensor::Exp_(){
        linalg::Exp_(*this);
        return *this;
    }
    Tensor Tensor::Exp(){
        return linalg::Exp(*this); 
    }


}//namespace cytnx


