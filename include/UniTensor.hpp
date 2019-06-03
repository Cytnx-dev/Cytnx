#ifndef _H_UniTensor_
#define _H_UniTensor_
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Storage.hpp"
#include "Device.hpp"
#include "Tensor.hpp"
#include "Symmetry.hpp"
#include "Bond.hpp"
#include "intrusive_ptr_base.hpp"
#include "utils/utils_internal_interface.hpp"
#include <iostream>
#include <vector>
#include <initializer_list>
#include "utils/utils.hpp"

namespace cytnx{
   
    /// @cond  
    class UniTensor_base: public intrusive_ptr_base<UniTensor_base>{
        protected:
            std::vector< Bond > _bonds;
            std::vector<cytnx_int64> _labels;
            bool _is_braket_form;
            bool _is_tag;
            cytnx_int64 _Rowrank;
            bool _is_diag;
            std::string _name;

            bool _update_braket(){
                if(_bonds.size()==0) return false;

                if(this->_bonds[0].type()!= bondType::BD_REG){
                    //check:
                    for(unsigned int i=0;i<this->_bonds.size();i++){
                        if(i<this->_Rowrank){
                            if(this->_bonds[i].type()!=bondType::BD_KET) return false;
                        }else{
                            if(this->_bonds[i].type()!=bondType::BD_BRA) return false;
                        }
                    }
                    return true;
                }else{
                    return false;
                }
            }


        public:
            friend class UniTensor; // allow wrapper to access the private elems
            UniTensor_base(): _is_tag(false), _name(std::string("")), _is_braket_form(false), _Rowrank(-1), _is_diag(false){};

            //copy&assignment constr., use intrusive_ptr's !!
            UniTensor_base(const UniTensor_base &rhs);
            UniTensor_base& operator=(UniTensor_base &rhs);

            cytnx_uint64 Rowrank() const{return this->_Rowrank;}
            bool is_diag() const{ return this->_is_diag; }
            const bool&     is_braket_form() const{
                return this->_is_braket_form;
            }
            const bool& is_tag() const{
                return this->_is_tag;
            }
            const std::vector<cytnx_int64>& labels() const{ return this->_labels;}
            const std::vector<Bond> &bonds() const {return this->_bonds;}       
            const std::string& name() const { return this->_name;}
            void set_name(const std::string &in){ this->_name = in;}

            virtual void Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels={}, const cytnx_int64 &Rowrank=-1,const unsigned int &dtype=Type.Double,const int &device = Device.cpu,const bool &is_diag=false);
            virtual std::vector<cytnx_uint64> shape() const;
            virtual bool      is_blockform() const ;
            virtual bool     is_contiguous() const;
            virtual void to_(const int &device);
            virtual boost::intrusive_ptr<UniTensor_base> to(const int &device) const;
            virtual boost::intrusive_ptr<UniTensor_base> clone() const;
            virtual unsigned int  dtype() const;
            virtual int          device() const;
            virtual std::string      dtype_str() const;
            virtual std::string     device_str() const;
            virtual boost::intrusive_ptr<UniTensor_base> permute(const std::vector<cytnx_int64> &mapper,const cytnx_int64 &Rowrank=-1, const bool &by_label=false);
            virtual void permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &Rowrank=-1, const bool &by_label=false);
            virtual void contiguous_();
            virtual boost::intrusive_ptr<UniTensor_base> contiguous();            
            virtual void print_diagram(const bool &bond_info=false);
            virtual Tensor get_block(const cytnx_uint64 &idx=0) const;
            virtual Tensor get_block(const std::vector<cytnx_int64> &qnum) const;
            virtual std::vector<Tensor> get_blocks() const;
    };
    /// @endcond

    //======================================================================
    /// @cond
    class DenseUniTensor: public UniTensor_base{
        protected:
            Tensor _block;
            DenseUniTensor* clone_meta() const{
                DenseUniTensor* tmp = new DenseUniTensor();
                tmp->_bonds = vec_clone(this->_bonds);
                tmp->_labels = this->_labels;
                tmp->_is_braket_form = this->_is_braket_form;
                tmp->_Rowrank = this->_Rowrank;
                tmp->_is_diag = this->_is_diag;
                tmp->_name = this->_name;
                tmp->_is_tag = this->_is_tag; 
                return tmp;
            }
        public:
            friend class UniTensor; // allow wrapper to access the private elems
            // virtual functions
            void Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels={}, const cytnx_int64 &Rowrank=-1, const unsigned int &dtype=Type.Double,const int &device = Device.cpu, const bool &is_diag=false);

            std::vector<cytnx_uint64> shape() const{ return this->_block.shape();}
            bool is_blockform() const{ return false;}
            void to_(const int &device){
                this->_block.to_(device);
            }
            boost::intrusive_ptr<UniTensor_base> to(const int &device) const{
                boost::intrusive_ptr<UniTensor_base> out = this->clone();
                out->to_(device);
                return out;    
            }
            boost::intrusive_ptr<UniTensor_base> clone() const{
                DenseUniTensor* tmp = this->clone_meta();
                tmp->_block = this->_block.clone();
                boost::intrusive_ptr<UniTensor_base> out(tmp);
                return out;
            };
            bool     is_contiguous() const{return this->_block.is_contiguous();}
            unsigned int  dtype() const{return this->_block.dtype();}
            int          device() const{return this->_block.device();}
            std::string      dtype_str() const{ return Type.getname(this->_block.dtype());}
            std::string     device_str() const{ return Device.getname(this->_block.device());}
            boost::intrusive_ptr<UniTensor_base> permute(const std::vector<cytnx_int64> &mapper,const cytnx_int64 &Rowrank=-1,const bool &by_label=false);
            void permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &Rowrank=-1, const bool &by_label=false);
            void contiguous_(){ this->_block.contiguous_();}
            boost::intrusive_ptr<UniTensor_base> contiguous(){
                // if contiguous then return self! 
                if(this->is_contiguous()){
                    boost::intrusive_ptr<UniTensor_base> out(this);
                    return out;
                }else{
                    DenseUniTensor* tmp = this->clone_meta();
                    tmp->_block = this->_block.contiguous();
                    boost::intrusive_ptr<UniTensor_base> out(tmp);
                    return out;
                }
            }
            void print_diagram(const bool &bond_info=false);         
            Tensor get_block(const cytnx_uint64 &idx=0) const{ return this->_block.clone(); }
            Tensor get_block(const std::vector<cytnx_int64> &qnum) const{return this->_block.clone();}
            std::vector<Tensor> get_blocks() const {
                std::vector<Tensor> out;
                out.push_back(this->_block.clone());
                return out; // this will share memory!!
            }
            // end virtual function              

    };
    /// @endcond


    //======================================================================
    /// @cond
    class SparseUniTensor: public UniTensor_base{
        protected:
            std::vector<Tensor> _blocks;
            std::vector<cytnx_uint64> _mapper;
            std::vector<cytnx_uint64> _inv_mapper;
            bool _contiguous;
            
            //boost::intrusive_ptr<UniTensor_base> clone_meta() const{};
        
        public:
            friend class UniTensor; // allow wrapper to access the private elems
            // virtual functions
            void Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels={}, const cytnx_int64 &Rowrank=-1, const unsigned int &dtype=Type.Double,const int &device = Device.cpu, const bool &is_diag=false);

            std::vector<cytnx_uint64> shape() const{ 
                std::vector<cytnx_uint64> out(this->_bonds.size());
                for(cytnx_uint64 i=0;i<out.size();i++){
                    out[i] = this->_bonds[i].dim();
                }
                return out;
            }
            bool is_blockform() const{return true;}
            void to_(const int &device){};
            boost::intrusive_ptr<UniTensor_base> to(const int &device) const{};
            boost::intrusive_ptr<UniTensor_base> clone() const{};
            bool     is_contiguous() const{};
            unsigned int  dtype() const{};
            int          device() const{};
            std::string      dtype_str() const{};
            std::string     device_str() const{};
            boost::intrusive_ptr<UniTensor_base> permute(const std::vector<cytnx_int64> &mapper,const cytnx_int64 &Rowrank=-1, const bool &by_label=false){};
            void permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &Rowrank=-1,const bool &by_label=false){};
            void contiguous_(){};
            boost::intrusive_ptr<UniTensor_base> contiguous(){};            
            void print_diagram(const bool &bond_info=false){};
            Tensor get_block(const cytnx_uint64 &idx=0) const{};
            Tensor get_block(const std::vector<cytnx_int64> &qnum) const{};
            std::vector<Tensor> get_blocks() const {};
            // end virtual func
    };
    /// @endcond

    //======================================================================
    //wrapper:
    class UniTensor{
        private:
            boost::intrusive_ptr<UniTensor_base> _impl;

        public:
            UniTensor(): _impl(new UniTensor_base()){};
            UniTensor(const UniTensor &rhs){
                this->_impl = rhs._impl;
            }            
            UniTensor& operator=(const UniTensor &rhs){
                this->_impl = rhs._impl;
                return *this;
            }

            UniTensor(const Tensor &in_tensor, const cytnx_uint64 &Rowrank){
                std::vector<Bond> bds;
                for(cytnx_uint64 i=0;i<in_tensor.shape().size();i++){
                    bds.push_back(Bond(in_tensor.shape()[i]));
                }
                DenseUniTensor* tmp = new DenseUniTensor();
                tmp->_block = in_tensor;
                tmp->_labels = utils_internal::range_cpu<cytnx_int64>(in_tensor.shape().size());
                tmp->_Rowrank = Rowrank;
                boost::intrusive_ptr<UniTensor_base> out(tmp);
                this->_impl = out;

            }


            UniTensor(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels={}, const cytnx_int64 &Rowrank=-1, const unsigned int &dtype=Type.Double, const int &device = Device.cpu, const bool &is_diag=false){
                this->Init(bonds,in_labels,Rowrank,dtype,device,is_diag);
            }

            void Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels={}, const cytnx_int64 &Rowrank=-1, const unsigned int &dtype=Type.Double, const int &device = Device.cpu, const bool &is_diag=false){

                // checking type:
                bool is_sym = false;
                for(cytnx_uint64 i=0;i<bonds.size();i++){
                    //check 
                    if(bonds[i].syms().size() != 0) is_sym = true;
                    else cytnx_error_msg(is_sym,"[ERROR] cannot have bonds with mixing of symmetry and non-symmetry.%s","\n");
                }

                // dynamical dispatch:
                if(is_sym){
                    boost::intrusive_ptr<UniTensor_base> out(new SparseUniTensor());
                    this->_impl = out;
                }else{   
                    boost::intrusive_ptr<UniTensor_base> out(new DenseUniTensor());
                    this->_impl = out;
                }
                this->_impl->Init(bonds, in_labels, Rowrank, dtype, device, is_diag);
            }
            
            void Init(const std::initializer_list<Bond> &bonds, const std::vector<cytnx_int64> &in_labels={}, const cytnx_int64 &Rowrank=-1,const unsigned int &dtype=Type.Double,const int &device = Device.cpu, const bool &is_diag=false){
                std::vector<Bond> vbonds = bonds;
                this->Init(vbonds,in_labels,Rowrank,dtype,device,is_diag);
            }
            void Init(const std::vector<Bond> &bonds, const std::initializer_list<cytnx_int64> &in_labels={}, const cytnx_int64 &Rowrank=-1, const unsigned int &dtype=Type.Double,const int &device = Device.cpu,const bool &is_diag=false){
                std::vector<cytnx_int64> vin_labels = in_labels;
                this->Init(bonds,vin_labels,Rowrank,dtype,device,is_diag);
            }
            void Init(const std::initializer_list<Bond> &bonds, const std::initializer_list<cytnx_int64> &in_labels={},const cytnx_int64 &Rowrank=-1, const unsigned int &dtype=Type.Double,const int &device = Device.cpu,const bool &is_diag=false){
                std::vector<Bond> vbonds = bonds;
                std::vector<cytnx_int64> vin_labels = in_labels;
                this->Init(vbonds,vin_labels,Rowrank,dtype,device,is_diag);
            }

            void set_name(const std::string &in){
                this->_impl->set_name(in);
            }

            cytnx_uint64 Rowrank() const{return this->_impl->Rowrank();}
            unsigned int  dtype() const{ return this->_impl->dtype(); }
            int          device() const{ return this->_impl->device();   }
            std::string name() const { return this->_impl->name();}
            std::string      dtype_str() const{ return this->_impl->dtype_str();}
            std::string     device_str() const{ return this->_impl->device_str();}
            bool     is_contiguous() const{ return this->_impl->is_contiguous();}
            bool is_diag() const{ return this->_impl->is_diag(); }
            bool is_tag() const { return this->_impl->is_tag();}
            const bool&     is_braket_form() const{
                return this->_impl->is_braket_form();
            }
            const std::vector<cytnx_int64>& labels() const{ return this->_impl->labels();}
            const std::vector<Bond> &bonds() const {return this->_impl->bonds();}       
            std::vector<cytnx_uint64> shape() const{return this->_impl->shape();}
            bool      is_blockform() const{ return this->_impl->is_blockform();}

            void to_(const int &device){this->_impl->to_(device);}
            UniTensor to(const int &device) const{ 
                UniTensor out;
                out._impl = this->_impl->to(device);
                return out;
            }
            UniTensor clone() const{
                UniTensor out;
                out._impl = this->_impl->clone();
                return out;
            }
            UniTensor permute(const std::vector<cytnx_int64> &mapper,const cytnx_int64 &Rowrank=-1,const bool &by_label=false){UniTensor out; out._impl = this->_impl->permute(mapper,Rowrank,by_label); return out;}
            void permute_(const std::vector<cytnx_int64> &mapper,const cytnx_int64 &Rowrank=-1,const bool &by_label=false){
                this->_impl->permute_(mapper,Rowrank,by_label);
            }
            UniTensor contiguous(){
                UniTensor out;
                out._impl = this->_impl->contiguous();
                return out;
            }
            void contiguous_(){
                this->_impl->contiguous_();
            }
            void print_diagram(const bool &bond_info=false){
               this->_impl->print_diagram(bond_info);
            }
            Tensor get_block(const cytnx_uint64 &idx=0) const{
                return this->_impl->get_block(idx);
            };
            Tensor get_block(const std::vector<cytnx_int64> &qnum) const{
                return this->_impl->get_block(qnum);
            }
            std::vector<Tensor> get_blocks() const {
                return this->_impl->get_blocks();
            }
    };

    std::ostream& operator<<(std::ostream& os, const UniTensor &in);

    // real implementation
    /// @cond
    /*
    class UniTensor_impl: public intrusive_ptr_base<UniTensor_impl>{
        private:
            std::vector< Tensor > _blocks;
            std::vector< Bond > _bonds;
            std::vector<cytnx_int64> _labels;
            std::vector<cytnx_int64> _total_qnums;

            // psudo-perm info
            std::vector< std::vector<cytnx_uint64> > _qnum_mapper;
            
            bool _contiguous;
            bool _braket;
            //bool _blockform;


            // these two are redundant for non-symmetry, but for block-symmetry, put_block can result as non-coalesence blocks on different devices. Hence we also trace the dype and device in here.
            unsigned int _dtype_id;
            int _device;
            cytnx_uint64 N_inbond;

        public:
            UniTensor_impl(): _braket(true),_contiguous(true), _device(-1), _dtype_id(0){};
            
            void Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels={}, const unsigned int &dtype=3, int device=-1, bool is_blockform=false){

                //check # of bond:
                cytnx_error_msg(bonds.size()==0,"%s","[ERROR] UniTensor must have at least one bond.");
                

                //check (non)Symmetry for all bonds
                vector<Symmetry> syms = bonds[0].get_syms();
                for(cytnx_uint32 i=0;i<bonds.size();i++){
                    if(bonds[i].get_type()==cytnx::BD_KET){ ket_idx.push_back(i); this->_bonds.push_back(bonds[i])};
                    else bra_idx.push_back(i);

                    //check 
                    cytnx_error_msg(syms.size()!=bonds[i].get_syms().size(),"%s","[ERROR] All bonds must have same symmetries.");
                    for(cytnx_uint32 j=0;j<syms.size();j++)
                        cytnx_error_msg(syms[j]!=bonds[i].get_syms()[j],"%s","[ERROR] All bonds mush have same symmetries.");
                }

                for(cytnx_uint32 j=0;j<bra_idx.size();j++)
                    this->_bonds.push_back(bonds[bra_idx[j]]);


                //check labels:
                if(in_labels.size()==0){
                    for(cytnx_int64 i=0;i<ket_idx.size();i++)
                        this->_labels.push_back(ket_idx[i]);    
                    for(cytnx_int64 i=0;i<bra_idx.size();i++)
                        this->_labels.push_back(bra_idx[i]);
                }else{
                    cytnx_error_msg(bonds.size()!=in_labels.size(),"%s","[ERROR] labels must have same lenth as # of bonds.");
                    for(cytnx_int64 i=0;i<ket_idx.size();i++)
                        this->_labels.push_back(in_labels[ket_idx[i]]);    
                    for(cytnx_int64 i=0;i<bra_idx.size();i++)
                        this->_labels.push_back(in_labels[bra_idx[i]]);
                }

                this->_device = device;
                this->dtype_id = dtype;

                if(syms.size()==0){

                    //non symmetry
                    std::vector<cytnx_uint64> _shape.resize(bonds.size());
                    for(unsigned int i=0;i<_shape.size();i++)
                        _shape.push_back(bonds[i].dim());
                    this->_blocks.push_back(Tensor(_shape,dtype,device));

                }else{
                    //symmetry
                    // check at-least one in-bond
                    cytnx_error_msg(1,"%s","Developing");
                }

                                
            }
            
            
            const std::vector<std::vector<cytnx_uint64> >& _get_mapper(){ return _mapper;}
            std::vector<Tensor>& _get_blocks(){ return this->_blocks;     }
            std::vector<Bond>&     get_bonds(){ return this->_bonds;      }   
            std::vector<cytnx_int64>& get_labels(){ return this->_labels; }

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
                    for(cytnx_uint32 i=0;i<this->_block.size();i++){
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
            T& at(const std::vector<cytnx_uint64> &locator){
                cytnx_error_msg(this->_total_qnums.size()!=0, "%s","UniTensor with symmetry should use GetBlock");
                cytnx_error_msg(locator.size() != this->_block[0]._shape.size(), "%s", "The input indexes rank is not match UniTensor's rank.");

                cytnx_uint64 RealRank,mtplyr;
                std::vector<cytnx_uint64> c_shape(this->_shape.size());
                std::vector<cytnx_uint64> c_loc(this->_shape.size());

                RealRank=0;
                mtplyr = 1;

                for(cytnx_int64 i=this->_shape.size()-1; i>=0; i--){
                    if(locator[i]>=this->_shape[i]){
                        cytnx_error_msg(true, "%s", "The dimension of rank that trying to access is exceed UniTensor's dimension.");
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

            
            //default device==Device.cpu (-1)
            void Init(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype, int device=-1){
                _impl->Init(shape,dtype,device);
            }
            void Init(const std::initializer_list<cytnx_uint64> &shape,const unsigned int &dtype, int device=-1){
                std::vector<cytnx_uint64> args = shape;
                _impl->Init(args,dtype,device);
            }

            unsigned int dtype_id(){return _impl->dtype_id();}
            int device_id(){ return this->_impl->device_id();}
            std::string dtype(){ return _impl->dtype();}
            std::string device(){ return this->_impl->device();}

            const std::vector<cytnx_uint64>& shape(){
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

            void permute_(const std::vector<cytnx_uint64> &rnks){
                this->_impl->permute_(rnks);
            }
            void permute_(const std::initializer_list<cytnx_uint64> &rnks){
                std::vector<cytnx_uint64> args = rnks;
                this->_impl->permute_(args);
            }

            UniTensor permute(const std::vector<cytnx_uint64> &rnks){
                UniTensor out;
                out._impl = this->_impl->permute(rnks);
                return out;
            }

            UniTensor permute(const std::initializer_list<cytnx_uint64> &rnks){
                UniTensor out;
                std::vector<cytnx_uint64> args = rnks;
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

            void Reshape_(const std::vector<cytnx_int64> &new_shape){
                this->_impl->Reshape_(new_shape);
            }

            void Reshape_(const std::initializer_list<cytnx_int64> &new_shape){
                std::vector<cytnx_int64> args = new_shape;
                this->_impl->Reshape_(args);
            }

            UniTensor Reshape(const std::vector<cytnx_int64> &new_shape){
                UniTensor out;
                out._impl = this->_impl->Reshape(new_shape);
                return out;
            }

            UniTensor Reshape(const std::initializer_list<cytnx_int64> &new_shape){
                std::vector<cytnx_int64> args = new_shape;
                return this->Reshape(args);
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
            template<class T>
            T& operator[](const std::vector<cytnx_uint64> &locator){
                return this->_impl->at<T>(locator);
            }
            template<class T>
            T& operator[](const std::initializer_list<cytnx_uint64> &locator){
                std::vector<cytnx_uint64> args = locator;
                return this->_impl->at<T>(args);
            }
    };// class UniTensor
    */
    /// @endcond
}

#endif
