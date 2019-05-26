#ifndef _H_Tensor_
#define _H_Tensor_
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Storage.hpp"
#include "Device.hpp"
#include "intrusive_ptr_base.hpp"
#include "utils/utils_internal.hpp"
#include <iostream>
#include <vector>
#include <initializer_list>

namespace cytnx{



    // real implementation
    class Tensor_impl: public intrusive_ptr_base<Tensor_impl>{
        private:

            //Interface:
            Storage_init_interface __SII;
            

            //Memory:
            boost::intrusive_ptr<Storage_base> _storage;

            //tensor shape
            std::vector<cytnx_uint64> _shape;

            // psudo-perm info
            std::vector<cytnx_uint64> _mapper;
            std::vector<cytnx_uint64> _invmapper;
            bool _contiguous;

        public:
            Tensor_impl():_storage(new Storage_base()), _contiguous(true){};
            
            void Init(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype, int device=-1);

            //copy&assignment constr., use intrusive_ptr's
            Tensor_impl(const Tensor_impl &rhs);
            Tensor_impl& operator=(const Tensor_impl &rhs); // add const
            
            const unsigned int dtype() const{
                return this->_storage->dtype;
            }
            const int device() const {
                return this->_storage->device;
            }

            const std::string dtype_str() const {
                return cytnxtype.getname(this->_storage->dtype);
            }
            const std::string device_str() const{
                return cytnxdevice.getname(this->_storage->device);
            }

            const std::vector<cytnx_uint64>& shape() const{
                return _shape;
            }
            const bool& is_contiguous() const{
                return this->_contiguous;
            }

            const std::vector<cytnx_uint64>& _get_mapper() const{
                return _mapper;
            }
            const std::vector<cytnx_uint64> & _get_invmapper() const{
                return _invmapper;
            }
            boost::intrusive_ptr<Storage_base>& _get_storage(){
                return _storage;
            }

            const boost::intrusive_ptr<Storage_base> _get_storage() const{
                return _storage;
            }

            boost::intrusive_ptr<Tensor_impl> copy(){
                boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());
                out->_storage = this->_storage->copy();
                out->_mapper = this->_mapper;
                out->_invmapper = this->_invmapper;
                out->_shape = this->_shape;
                out->_contiguous = this->_contiguous;
                return out;
            }

            void to_(const int &device){
                this->_storage->to_(device);
            }
            boost::intrusive_ptr<Tensor_impl> to(const int &device){
                if(this->device()==device){
                    return this;
                }else{
                    boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());
                    out->_storage = this->_storage->to(device);
                    out->_mapper = this->_mapper;
                    out->_invmapper = this->_invmapper;
                    out->_shape = this->_shape;
                    out->_contiguous = this->_contiguous;
                    return out;
                }
            }

            void permute_(const std::vector<cytnx_uint64> &rnks);


            boost::intrusive_ptr<Tensor_impl> permute(const std::vector<cytnx_uint64> &rnks){
                boost::intrusive_ptr<Tensor_impl> out = this->copy();
                out->permute_(rnks);
                return out;
            }            
 
            template<class T> 
            T& at(const std::vector<cytnx_uint64> &locator){
                cytnx_error_msg(locator.size() != this->_shape.size(), "%s", "The input indexes rank is not match Tensor's rank.");

                cytnx_uint64 RealRank,mtplyr;
                std::vector<cytnx_uint64> c_shape(this->_shape.size());
                std::vector<cytnx_uint64> c_loc(this->_shape.size());

                RealRank=0;
                mtplyr = 1;

                for(cytnx_int64 i=this->_shape.size()-1; i>=0; i--){
                    if(locator[i]>=this->_shape[i]){
                        cytnx_error_msg(true, "%s", "The dimension of rank that trying to access is exceed Tensor's dimension.");
                    }
                    c_shape[i] = this->_shape[this->_invmapper[i]];
                    c_loc[i] = locator[this->_invmapper[i]];
                    RealRank += mtplyr*c_loc[i];
                    mtplyr *= c_shape[i];
                }
                return this->_storage->at<T>(RealRank);
            }

            boost::intrusive_ptr<Tensor_impl> Contiguous(){
                // return new instance if act on non-contiguous tensor
                // return self if act on contiguous tensor
                if(this->_contiguous){
                    boost::intrusive_ptr<Tensor_impl> out(this);
                    return out;
                }else{
                    boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());
                    std::vector<cytnx_uint64> oldshape(this->_shape.size());
                    for(cytnx_uint64 i=0;i<this->_shape.size();i++){
                        oldshape[i] = this->_shape[this->_invmapper[i]];
                    }
        
                    out->_storage = this->_storage->Move_memory(oldshape,this->_mapper, this->_invmapper);
                    out->_invmapper = utils_internal::range_cpu(this->_invmapper.size());
                    out->_mapper = out->_invmapper;
                    out->_shape = this->_shape;
                    out->_contiguous = true;
                    return out;
                }
            }
            
            void Contiguous_(){
                // return new instance if act on non-contiguous tensor
                // return self if act on contiguous tensor
                if(!this->_contiguous){
                    std::vector<cytnx_uint64> oldshape(this->_shape.size());
                    for(cytnx_uint64 i=0;i<this->_shape.size();i++){
                        oldshape[i] = this->_shape[this->_invmapper[i]];
                    }
                    this->_storage->Move_memory_(oldshape,this->_mapper, this->_invmapper);
                    this->_mapper = utils_internal::range_cpu(this->_invmapper.size());
                    this->_invmapper = this->_mapper;
                    this->_contiguous = true;
                }
            }

            void Reshape_(const std::vector<cytnx_int64> &new_shape){
                if(!this->_contiguous){
                    this->Contiguous_();
                }
                std::vector<cytnx_uint64> result_shape(new_shape.size());
                cytnx_uint64 new_N = 1;
                bool has_undetermine = false;
                unsigned int Udet_id = 0;
                for(int i=0;i<new_shape.size();i++){
                    if(new_shape[i]<0){
                        if(new_shape[i]!=-1) cytnx_error_msg(new_shape[i]!=-1,"%s","[ERROR] Reshape can only have dimension > 0 and one undetermine rank specify as -1");
                        if(has_undetermine) cytnx_error_msg(new_shape[i]!=-1,"%s","[ERROR] Reshape can only have dimension > 0 and one undetermine rank specify as -1");
                        Udet_id = i;
                        has_undetermine = true;
                    }else{
                        new_N *= new_shape[i];
                        result_shape[i] = new_shape[i];
                    }
                }

                            
                if(has_undetermine){
                    cytnx_error_msg(new_N >= this->_storage->len,"%s","[ERROR] new shape exceed the total number of elements.");
                    cytnx_error_msg(this->_storage->len%new_N,"%s","[ERROR] unmatch size when reshape with undetermine dimension");
                    result_shape[Udet_id] = this->_storage->len/new_N;
                }else{
                    cytnx_error_msg(new_N != this->_storage->len,"%s","[ERROR] new shape does not match the number of elements.");
                }
            
                this->_shape = result_shape;
                this->_mapper = utils_internal::range_cpu(result_shape.size());
                this->_invmapper = this->_mapper; 
            }


            boost::intrusive_ptr<Tensor_impl> Reshape(const std::vector<cytnx_int64> &new_shape){
                boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());
                if(!this->_contiguous){
                    out = this->Contiguous();
                }else{
                    out = this->copy();
                }

                out->Reshape_(new_shape);
                return out;
            }


            
            boost::intrusive_ptr<Tensor_impl> astype(const int& new_type) const {
                boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());
                out->_storage = this->_storage->astype(new_type);
                return out;
            }

    };


    // wrapping around, API
    class Tensor{
        private:
        public:

            boost::intrusive_ptr<Tensor_impl> _impl;
            Tensor():_impl(new Tensor_impl()){};
            Tensor(const Tensor &rhs){
                _impl = rhs._impl;
            }
            Tensor& operator=(const Tensor &rhs){
                _impl = rhs._impl;
            }
             
            //default device==cytnxdevice.cpu (-1)
            void Init(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype, int device=-1){
                _impl->Init(shape,dtype,device);
            }
            void Init(const std::initializer_list<cytnx_uint64> &shape,const unsigned int &dtype, int device=-1){
                std::vector<cytnx_uint64> args = shape;
                _impl->Init(args,dtype,device);
            }

            Tensor(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype, int device=-1): _impl(new Tensor_impl()){
                this->Init(shape,dtype,device);
            }
            Tensor(const std::initializer_list<cytnx_uint64> &shape, const unsigned int &dtype,int device=-1): _impl(new Tensor_impl()){
                this->Init(shape,dtype,device);
            }

            const unsigned int dtype() const {return this->_impl->dtype();}
            const int device() const { return this->_impl->device();}
            const std::string dtype_str() const { return this->_impl->dtype_str();}
            const std::string device_str() const{ return this->_impl->device_str();}

            const std::vector<cytnx_uint64>& shape() const{
                return this->_impl->shape();
            }

            Tensor copy(){
                Tensor out;
                out._impl = this->_impl->copy();
                return out;
            }
            Tensor to(const int &device){
                Tensor out;
                out._impl = this->_impl->to(device);
                return out;
            }
            void to_(const int &device){
                this->_impl->to_(device);
            }
            
            const bool& is_contiguous() const{
                return this->_impl->is_contiguous();
            }

            void permute_(const std::vector<cytnx_uint64> &rnks){
                this->_impl->permute_(rnks);
            }
            void permute_(const std::initializer_list<cytnx_uint64> &rnks){
                std::vector<cytnx_uint64> args = rnks;
                this->_impl->permute_(args);
            }

            Tensor permute(const std::vector<cytnx_uint64> &rnks){
                Tensor out;
                out._impl = this->_impl->permute(rnks);
                return out;
            }

            Tensor permute(const std::initializer_list<cytnx_uint64> &rnks){
                Tensor out;
                std::vector<cytnx_uint64> args = rnks;
                out._impl = this->_impl->permute(args);
                return out;
            }

            Tensor Contiguous(){
                Tensor out;
                out._impl = this->_impl->Contiguous();
                return out;
            }
            void Contiguous_(){
                this->_impl->Contiguous_();
            }

            void Reshape_(const std::vector<cytnx_int64> &new_shape){
                this->_impl->Reshape_(new_shape);
            }

            void Reshape_(const std::initializer_list<cytnx_int64> &new_shape){
                std::vector<cytnx_int64> args = new_shape;
                this->_impl->Reshape_(args);
            }

            Tensor Reshape(const std::vector<cytnx_int64> &new_shape){
                Tensor out;
                out._impl = this->_impl->Reshape(new_shape);
                return out;
            }

            Tensor Reshape(const std::initializer_list<cytnx_int64> &new_shape){
                std::vector<cytnx_int64> args = new_shape;
                return this->Reshape(args);
            }


            Tensor astype(const int &new_type) const{
                Tensor out;
                out._impl = this->_impl->astype(new_type);
                return out;
            }

            template<class T>
            T& at(const std::vector<cytnx_uint64> &locator){
                return this->_impl->at<T>(locator);
            }
            template<class T>
            T& at(const std::initializer_list<cytnx_uint64> &locator){
                std::vector<cytnx_uint64> args = locator;
                return this->_impl->at<T>(args);
            }


            // Arithmic:
            /*           
           template<class T>
           Tensor& operator+=(const T &rc){
                *this = cytnx::linalg::Add(*this,rc);
                return *this;
           }
           
           
           template<class T>
           Tensor Add(const T &rhs){
                return cytnx::linalg::Add(*this,rhs); 
           }
           template<class T>
           Tensor& Add_(const T &rhs){
                *this = cytnx::linalg::Add(*this,rhs);
                return *this;
           }
           
           
           template<class T>
           Tensor& operator-=(const T &rc){
                *this = cytnx::linalg::Sub(*this,rc);
                return *this;
           } 
           
           
           template<class T>
           Tensor Sub(const T &rhs){
                return cytnx::linalg::Sub(*this,rhs); 
           }
           template<class T>
           Tensor& Sub_(const T &rhs){
                *this = cytnx::linalg::Sub(*this,rhs);
                return *this;
           }
           
           
           template<class T>
           Tensor& operator*=(const T &rc){
                *this = cytnx::linalg::Mul(*this,rc);
                return *this;
           } 
           
           
           template<class T>
           Tensor Mul(const T &rhs){
                return cytnx::linalg::Mul(*this,rhs); 
           }
           template<class T>
           Tensor& Mul_(const T &rhs){
                *this = cytnx::linalg::Mul(*this,rhs);
                return *this;
           }
           
           
           template<class T>
           Tensor& operator/=(const T &rc){
                *this = cytnx::linalg::Div(*this,rc);
                return *this;
           }
            
           
           template<class T>
           Tensor Div(const T &rhs){
                return cytnx::linalg::Div(*this,rhs); 
           }
           template<class T>
           Tensor& Div_(const T &rhs){
                *this = cytnx::linalg::Div(*this,rhs);
                return *this;
           }
           */
            
            // linalg:
            //std::vector<Tensor> Svd(const bool &is_U, const bool &is_vT){
            //    return linalg::Svd(*this,is_U,is_vT);
            //}
            



    };// class Tensor

    std::ostream& operator<<(std::ostream& os, Tensor &in);

}

#endif
