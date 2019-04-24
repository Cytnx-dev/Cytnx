#ifndef _H_UniTensor_
#define _H_UniTensor_
#include "Type.hpp"
#include "tor10_error.hpp"
#include "Storage.hpp"
#include "Device.hpp"
#include "intrusive_ptr_base.hpp"
#include "utils/utils_internal.hpp"
#include <iostream>
#include <vector>
#include <initializer_list>

namespace tor10{


    // real implementation
    class UniTensor_impl: public intrusive_ptr_base<UniTensor_impl>{
        private:
            std::vector< Tensor > _blocks;
            std::vector< Bond > _bonds;
            std::vector<tor10_int64> _labels;
            std::vector<tor10_int64> _total_qnums;

            // psudo-perm info
            std::vector< std::vector<tor10_uint64> > _qnum_mapper;
            
            bool _contiguous;
            bool _braket;
            //bool _blockform;


            // these two are redundant for non-symmetry, but for block-symmetry, put_block can result as non-coalesence blocks on different devices. Hence we also trace the dype and device in here.
            unsigned int _dtype_id;
            int _device;
            tor10_uint64 N_inbond;

        public:
            UniTensor_impl(): _braket(true),_contiguous(true), _device(-1), _dtype_id(0){};
            
            void Init(const std::vector<Bond> &bonds, const std::vector<tor10_int64> &in_labels={}, const unsigned int &dtype=3, int device=-1, bool is_blockform=false){

                //check # of bond:
                tor10_error_msg(bonds.size()==0,"%s","[ERROR] UniTensor must have at least one bond.");
                

                //check (non)Symmetry for all bonds
                vector<Symmetry> syms = bonds[0].get_syms();
                for(tor10_uint32 i=0;i<bonds.size();i++){
                    if(bonds[i].get_type()==tor10::BD_KET){ ket_idx.push_back(i); this->_bonds.push_back(bonds[i])};
                    else bra_idx.push_back(i);

                    //check 
                    tor10_error_msg(syms.size()!=bonds[i].get_syms().size(),"%s","[ERROR] All bonds must have same symmetries.");
                    for(tor10_uint32 j=0;j<syms.size();j++)
                        tor10_error_msg(syms[j]!=bonds[i].get_syms()[j],"%s","[ERROR] All bonds mush have same symmetries.");
                }

                for(tor10_uint32 j=0;j<bra_idx.size();j++)
                    this->_bonds.push_back(bonds[bra_idx[j]]);


                //check labels:
                if(in_labels.size()==0){
                    for(tor10_int64 i=0;i<ket_idx.size();i++)
                        this->_labels.push_back(ket_idx[i]);    
                    for(tor10_int64 i=0;i<bra_idx.size();i++)
                        this->_labels.push_back(bra_idx[i]);
                }else{
                    tor10_error_msg(bonds.size()!=in_labels.size(),"%s","[ERROR] labels must have same lenth as # of bonds.");
                    for(tor10_int64 i=0;i<ket_idx.size();i++)
                        this->_labels.push_back(in_labels[ket_idx[i]]);    
                    for(tor10_int64 i=0;i<bra_idx.size();i++)
                        this->_labels.push_back(in_labels[bra_idx[i]]);
                }

                this->_device = device;
                this->dtype_id = dtype;

                if(syms.size()==0){

                    //non symmetry
                    std::vector<tor10_uint64> _shape.resize(bonds.size());
                    for(unsigned int i=0;i<_shape.size();i++)
                        _shape.push_back(bonds[i].dim());
                    this->_blocks.push_back(Tensor(_shape,dtype,device));

                }else{
                    //symmetry
                    // check at-least one in-bond
                    tor10_error_msg(1,"%s","Developing");
                }

                                
            }
            
            void Init(const std::intializer_list<Bond> &bonds, const std::vector<tor10_int64> &in_labels={}, const unsigned int &dtype=3, int device=-1){
                std::vector<Bond> args = bonds;
                this->Init(args,in_labels,dtype,device);
            }

            void Init(const std::vector<Bond> &bonds, const std::initializer_list<tor10_int64> &in_labels={}, const unsigned int &dtype=3, int device=-1){
                std::vector<tor10_int64> args = in_labels;
                this->Init(bonds,args,dtype,device);
            }

            void Init(const std::initializer_list<Bond> &bonds, const std::initializer_list<tor10_int64> &in_labels={},const unsigned int &dtype=3, int device=-1){
                std::vector<Bond> args_b = bonds;
                std::vector<tor10_int64> args_l = in_labels;
                this->Init(args_b,args_l,dtype,device);
            }


            //copy&assignment constr., use intrusive_ptr's !!
            UniTensor_impl(const UniTensor_impl &rhs);
            UniTensor_impl& operator=(UniTensor_impl &rhs);
            
            const unsigned int&  dtype_id(){ return this->_dtype_id; }
            const int&          device_id(){ return this->_device;   }
            std::string             dtype(){ return tor10type.getname(this->_dtype_id);}
            std::string            device(){ return tor10device.getname(this->_device);}

            const bool&     is_contiguous(){ return this->_contiguous;}
            const bool&      is_blockform(){ return this->_blockform; }
            const std::vector<std::vector<tor10_uint64> >& _get_mapper(){ return _mapper;}
            std::vector<Tensor>& _get_blocks(){ return this->_blocks;     }
            std::vector<Bond>&     get_bonds(){ return this->_bonds;      }   
            std::vector<tor10_int64>& get_labels(){ return this->_labels; }

            boost::intrusive_ptr<UniTensor_impl> copy(){

                boost::intrusive_ptr<UniTensor_impl> out(new UniTensor_impl());
                out->_blocks = vector<Tensor>(len(this->_blocks.size()));
                for(int i=0;i<out->_blocks.size();i++)
                    out->blocks[i] = this->_blocks[i].copy();

                out->_mapper = this->_mapper;
                out->_contiguous = this->_contiguous;
                out->_blockform = this->_blockform;
                out->_bonds  = this->_bonds;
                out->_dtype_id = this->_dtype_id;
                out->_device = this->_device;
                return out;
            }

            void to_(const int &device){
                if(this->_device!=device){
                    for(tor10_uint32 i=0;i<this->_block.size();i++){
                        this->_block[i].to_(device);
                    }
                    this->_device = device;
                }
            }

            boost::intrusive_ptr<UniTensor_impl> to(const int &device){
                if(this->device!=device){
                    boost::intrusive_ptr<UniTensor_impl> out(new UniTensor_impl());
                    out->_mapper = this->_mapper;
                    out->_contiguous = this->_contiguous;
                    out->_blockform = this->_blockform;
                    out->_bonds  = this->_bonds;
                    out->_dtype_id = this->_dtype_id;
                    out->_device = this->_device;
                    out->_block.resize(this->_block.size());
                    for(int i=0;i<out->_blocks.size();i++)
                        out->blocks[i] = this->_blocks[i].to(device);
                    return out;
                }else{
                    return this;
                }
            }

 
            template<class T> 
            T& at(const std::vector<tor10_uint64> &locator){
                tor10_error_msg(this->_total_qnums.size()!=0, "%s","UniTensor with symmetry should use GetBlock");
                tor10_error_msg(locator.size() != this->_block[0]._shape.size(), "%s", "The input indexes rank is not match UniTensor's rank.");

                tor10_uint64 RealRank,mtplyr;
                std::vector<tor10_uint64> c_shape(this->_shape.size());
                std::vector<tor10_uint64> c_loc(this->_shape.size());

                RealRank=0;
                mtplyr = 1;

                for(tor10_int64 i=this->_shape.size()-1; i>=0; i--){
                    if(locator[i]>=this->_shape[i]){
                        tor10_error_msg(true, "%s", "The dimension of rank that trying to access is exceed UniTensor's dimension.");
                    }
                    c_shape[i] = this->_shape[this->_invmapper[i]];
                    c_loc[i] = locator[this->_invmapper[i]];
                    RealRank += mtplyr*c_loc[i];
                    mtplyr *= c_shape[i];
                }
                return this->_storage->at<T>(RealRank);
            }

            boost::intrusive_ptr<UniTensor_impl> Contiguous(){
                // return new instance if act on non-contiguous tensor
                // return self if act on contiguous tensor
                if(this->_contiguous){
                    boost::intrusive_ptr<UniTensor_impl> out(this);
                    return out;
                }else{
                    boost::intrusive_ptr<UniTensor_impl> out(new UniTensor_impl());
                    std::vector<tor10_uint64> oldshape(this->_shape.size());
                    for(tor10_uint64 i=0;i<this->_shape.size();i++){
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
                    std::vector<tor10_uint64> oldshape(this->_shape.size());
                    for(tor10_uint64 i=0;i<this->_shape.size();i++){
                        oldshape[i] = this->_shape[this->_invmapper[i]];
                    }
                    this->_storage->Move_memory_(oldshape,this->_mapper, this->_invmapper);
                    this->_mapper = utils_internal::range_cpu(this->_invmapper.size());
                    this->_invmapper = this->_mapper;
                    this->_contiguous = true;
                }
            }


    };


    // wrapping around, API
    class UniTensor{
        private:
        public:

            boost::intrusive_ptr<UniTensor_impl> _impl;
            UniTensor():_impl(new UniTensor_impl()){};
            UniTensor(const UniTensor &rhs){
                _impl = rhs._impl;
            }
            UniTensor& operator=(UniTensor &rhs){
                _impl = rhs._impl;
            }

            
            //default device==tor10device.cpu (-1)
            void Init(const std::vector<tor10_uint64> &shape, const unsigned int &dtype, int device=-1){
                _impl->Init(shape,dtype,device);
            }
            void Init(const std::initializer_list<tor10_uint64> &shape,const unsigned int &dtype, int device=-1){
                std::vector<tor10_uint64> args = shape;
                _impl->Init(args,dtype,device);
            }

            unsigned int dtype_id(){return _impl->dtype_id();}
            int device_id(){ return this->_impl->device_id();}
            std::string dtype(){ return _impl->dtype();}
            std::string device(){ return this->_impl->device();}

            const std::vector<tor10_uint64>& shape(){
                return this->_impl->shape();
            }

            UniTensor copy(){
                UniTensor out;
                out._impl = this->_impl->copy();
                return out;
            }
            UniTensor to(const int &device){
                UniTensor out;
                out._impl = this->_impl->to(device);
                return out;
            }
            void to_(const int &device){
                this->_impl->to_(device);
            }
            
            const bool& is_contiguous(){
                return this->_impl->is_contiguous();
            }

            void permute_(const std::vector<tor10_uint64> &rnks){
                this->_impl->permute_(rnks);
            }
            void permute_(const std::initializer_list<tor10_uint64> &rnks){
                std::vector<tor10_uint64> args = rnks;
                this->_impl->permute_(args);
            }

            UniTensor permute(const std::vector<tor10_uint64> &rnks){
                UniTensor out;
                out._impl = this->_impl->permute(rnks);
                return out;
            }

            UniTensor permute(const std::initializer_list<tor10_uint64> &rnks){
                UniTensor out;
                std::vector<tor10_uint64> args = rnks;
                out._impl = this->_impl->permute(args);
                return out;
            }

            UniTensor Contiguous(){
                UniTensor out;
                out._impl = this->_impl->Contiguous();
                return out;
            }
            void Contiguous_(){
                this->_impl->Contiguous_();
            }

            void Reshape_(const std::vector<tor10_int64> &new_shape){
                this->_impl->Reshape_(new_shape);
            }

            void Reshape_(const std::initializer_list<tor10_int64> &new_shape){
                std::vector<tor10_int64> args = new_shape;
                this->_impl->Reshape_(args);
            }

            UniTensor Reshape(const std::vector<tor10_int64> &new_shape){
                UniTensor out;
                out._impl = this->_impl->Reshape(new_shape);
                return out;
            }

            UniTensor Reshape(const std::initializer_list<tor10_int64> &new_shape){
                std::vector<tor10_int64> args = new_shape;
                return this->Reshape(args);
            }


            template<class T>
            T& at(const std::vector<tor10_uint64> &locator){
                return this->_impl->at<T>(locator);
            }
            template<class T>
            T& at(const std::initializer_list<tor10_uint64> &locator){
                std::vector<tor10_uint64> args = locator;
                return this->_impl->at<T>(args);
            }
            template<class T>
            T& operator[](const std::vector<tor10_uint64> &locator){
                return this->_impl->at<T>(locator);
            }
            template<class T>
            T& operator[](const std::initializer_list<tor10_uint64> &locator){
                std::vector<tor10_uint64> args = locator;
                return this->_impl->at<T>(args);
            }
    };// class UniTensor

    std::ostream& operator<<(std::ostream& os, UniTensor &in);

}

#endif
