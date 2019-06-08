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
#include "utils/utils.hpp"
#include <iostream>
#include <vector>
#include <initializer_list>
#include <algorithm>
namespace cytnx{
   
    /// @cond 
    //class DenseUniTensor;
    //class SparseUniTensor; 
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
            void set_Rowrank(const cytnx_uint64 &new_Rowrank){
                cytnx_error_msg(new_Rowrank >= this->_labels.size(),"[ERROR] Rowrank cannot exceed the rank of UniTensor.%s","\n");
                this->_Rowrank = new_Rowrank;
            }
            void set_label(const cytnx_uint64 &idx, const cytnx_int64 &new_label){
                cytnx_error_msg(idx>=this->_labels.size(),"[ERROR] index exceed the rank of UniTensor%s","\n");
                //check in:
                bool is_dup =false;
                for(cytnx_uint64 i=0;i<this->_labels.size();i++){
                    if(new_label == this->_labels[i]){is_dup = true; break;}
                }
                cytnx_error_msg(is_dup,"[ERROR] alreay has a label that is the same as the input label%s","\n");
                this->_labels[idx] = new_label;                
            }
            void set_labels(const std::vector<cytnx_int64> &new_labels);
            template<class T>
            T& at(const std::vector<cytnx_uint64> &locator){
                cytnx_error_msg(this->is_blockform(),"[ERROR] cannot access element using at<T> on a UniTensor with symmetry.\n suggestion: get_block/get_blocks first.%s","\n");
                
                // only non-symm can enter this
                return this->get_block_().at<T>(locator);
            }
            


            virtual void Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels={}, const cytnx_int64 &Rowrank=-1,const unsigned int &dtype=Type.Double,const int &device = Device.cpu,const bool &is_diag=false);
            virtual void Init_by_Tensor(const Tensor& in, const cytnx_uint64 &Rowrank);
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
            virtual Tensor get_block(const cytnx_uint64 &idx=0) const; // return a copy of block
            virtual Tensor get_block(const std::vector<cytnx_int64> &qnum) const; //return a copy of block
            virtual Tensor get_block_(const cytnx_uint64 &idx=0) const; // return a share view of block, this only work for non-symm tensor.
            virtual std::vector<Tensor> get_blocks() const;
            virtual void put_block(const Tensor &in, const cytnx_uint64 &idx=0);
            virtual void put_block(const Tensor &in, const std::vector<cytnx_int64> &qnum);

            // this will only work on non-symm tensor (DenseUniTensor)
            virtual boost::intrusive_ptr<UniTensor_base> get(const std::vector<Accessor> &accessors);
            // this will only work on non-symm tensor (DenseUniTensor)
            virtual void set(const std::vector<Accessor> &accessors, const Tensor &rhs);
            virtual void reshape_(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &Rowrank=0);
            virtual boost::intrusive_ptr<UniTensor_base> reshape(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &Rowrank=0);
            virtual boost::intrusive_ptr<UniTensor_base> to_dense();
            virtual void to_dense_();
            virtual void combineBonds(const std::vector<cytnx_int64> &indicators, const bool &by_label=true);
            virtual ~UniTensor_base(){};
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
            // this only work for non-symm tensor
            void Init_by_Tensor(const Tensor& in_tensor, const cytnx_uint64 &Rowrank){
                cytnx_error_msg(in_tensor.dtype() == Type.Void,"[ERROR][Init_by_Tensor] cannot init a UniTensor from an un-initialize Tensor.%s","\n");
                if(in_tensor.storage().size() == 1){
                    //scalalr:
                    cytnx_error_msg(Rowrank != 0, "[ERROR][Init_by_Tensor] detect the input Tensor is a scalar with only one element. the Rowrank should be =0%s","\n");
                    this->_bonds.clear();
                    this->_block = in_tensor;
                    this->_labels.clear();
                    this->_Rowrank = Rowrank;
                }else{
                    std::vector<Bond> bds;
                    for(cytnx_uint64 i=0;i<in_tensor.shape().size();i++){
                        bds.push_back(Bond(in_tensor.shape()[i]));
                    }
                    this->_bonds = bds;
                    this->_block = in_tensor;
                    this->_labels = utils_internal::range_cpu<cytnx_int64>(in_tensor.shape().size());
                    cytnx_error_msg(Rowrank > in_tensor.shape().size(),"[ERROR][Init_by_tensor] Rowrank exceed the rank of Tensor.%s","\n");
                    this->_Rowrank = Rowrank;
                }
            }
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
            Tensor get_block(const std::vector<cytnx_int64> &qnum) const{cytnx_error_msg(true,"[ERROR][DenseUniTensor] try to get_block using qnum on a non-symmetry UniTensor%s","\n"); return Tensor();}
            std::vector<Tensor> get_blocks() const {
                std::vector<Tensor> out;
                out.push_back(this->_block.clone());
                return out; // this will share memory!!
            }
            // return a share view of block, this only work for non-symm tensor.
            Tensor get_block_(const cytnx_uint64 &idx=0) const{
                return this->_block;
            }

            void put_block(const Tensor &in, const cytnx_uint64 &idx=0){
                cytnx_error_msg(in.shape() != this->shape(),"[ERROR][DenseUniTensor] put_block, the input tensor shape does not match.%s","\n");
                this->_block = in;
            }
            void put_block(const Tensor &in, const std::vector<cytnx_int64> &qnum){
                cytnx_error_msg(true,"[ERROR][DenseUniTensor] try to put_block using qnum on a non-symmetry UniTensor%s","\n");
            }
            // this will only work on non-symm tensor (DenseUniTensor)
            boost::intrusive_ptr<UniTensor_base> get(const std::vector<Accessor> &accessors){
                boost::intrusive_ptr<UniTensor_base> out(new DenseUniTensor());
                out->Init_by_Tensor(this->_block.get(accessors),0); //wrapping around. 
                return out;
            }
            // this will only work on non-symm tensor (DenseUniTensor)
            void set(const std::vector<Accessor> &accessors, const Tensor &rhs){
                this->_block.set(accessors,rhs);
            }
            void reshape_(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &Rowrank=0){
                cytnx_error_msg(this->is_tag(),"[ERROR] cannot reshape a tagged UniTensor. suggestion: use untag() first.%s","\n");
                cytnx_error_msg(Rowrank > new_shape.size(), "[ERROR] Rowrank cannot larger than the rank of reshaped UniTensor.%s","\n");
                this->_block.reshape_(new_shape);
                this->Init_by_Tensor(this->_block,Rowrank);
            }
            boost::intrusive_ptr<UniTensor_base> reshape(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &Rowrank=0){
                cytnx_error_msg(this->is_tag(),"[ERROR] cannot reshape a tagged UniTensor. suggestion: use untag() first.%s","\n");
                cytnx_error_msg(Rowrank > new_shape.size(), "[ERROR] Rowrank cannot larger than the rank of reshaped UniTensor.%s","\n");
                boost::intrusive_ptr<UniTensor_base> out(new DenseUniTensor());
                out->Init_by_Tensor(this->_block.reshape(new_shape),Rowrank);
                return out;
            }
            boost::intrusive_ptr<UniTensor_base> to_dense(){
                cytnx_error_msg(!(this->_is_diag),"[ERROR] to_dense can only operate on UniTensor with is_diag = True.%s","\n");
                DenseUniTensor *tmp = this->clone_meta();
                tmp->_block = cytnx::linalg::Diag(this->_block);
                tmp->_is_diag = false;
                boost::intrusive_ptr<UniTensor_base> out(tmp);
                return out;
            }
            void to_dense_(){
                cytnx_error_msg(!(this->_is_diag),"[ERROR] to_dense_ can only operate on UniTensor with is_diag = True.%s","\n");
                this->_block = cytnx::linalg::Diag(this->_block);
                this->_is_diag = false;
            }

            void combineBonds(const std::vector<cytnx_int64> &indicators, const bool &by_label=true){
                cytnx_error_msg(indicators.size() < 2,"[ERROR] the number of bonds to combine must be > 1%s","\n");
                std::vector<cytnx_int64>::iterator it;
                std::vector<cytnx_int64> idx_mapper;
                if(by_label){
                    
                    //find the index of label:
                    for(cytnx_uint64 i=0;i<indicators.size();i++){
                        it = std::find(this->_labels.begin(),this->_labels.end(),indicators[i]);
                        cytnx_error_msg(it == this->_labels.end(),"[ERROR] labels not found in current UniTensor%s","\n");
                        idx_mapper.push_back(std::distance(this->_labels.begin(),it));
                    }

                }else{
                    idx_mapper = indicators;
                }

                ///first permute the Tensor:
                std::vector<cytnx_uint64> old_shape = this->shape();
                
                cytnx_error_msg(this->_is_diag,"[ERROR] cannot combineBond on a is_diag=True UniTensor. suggestion: try UniTensor.to_dense()/to_dense_() first.%s","\n");
                 


            }

            ~DenseUniTensor(){};
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
            void Init_by_Tensor(const Tensor& in_tensor, const cytnx_uint64 &Rowrank){
                cytnx_error_msg(true,"[ERROR][SparseUniTensor] cannot use Init_by_tensor() on a SparseUniTensor.%s","\n");
            }
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
            // return a share view of block, this only work for non-symm tensor.
            Tensor get_block_(const cytnx_uint64 &idx=0) const{
                cytnx_error_msg(true,"[ERROR][SparseUniTensor] cannot use get_block_() on a UniTensor with symmetry.\n suggestion: try get_block()/get_blocks()%s","\n");
                return Tensor();
            }
            std::vector<Tensor> get_blocks() const {};
            void put_block(const Tensor &in,const cytnx_uint64 &idx=0){};
            void put_block(const Tensor &in, const std::vector<cytnx_int64> &qnum){};
            // this will only work on non-symm tensor (DenseUniTensor)
            boost::intrusive_ptr<UniTensor_base> get(const std::vector<Accessor> &accessors){
                cytnx_error_msg(true,"[ERROR][SparseUniTensor][get] cannot use get on a UniTensor with Symmetry.\n suggestion: try get_block()/get_blocks() first.%s","\n");
                  
            }
            // this will only work on non-symm tensor (DenseUniTensor)
            void set(const std::vector<Accessor> &accessors, const Tensor &rhs){
                cytnx_error_msg(true,"[ERROR][SparseUniTensor][set] cannot use set on a UniTensor with Symmetry.\n suggestion: try get_block()/get_blocks() first.%s","\n");
            }
            void reshape_(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &Rowrank=0){
                cytnx_error_msg(true,"[ERROR] cannot reshape a UniTensor with symmetry.%s","\n");
            }
            boost::intrusive_ptr<UniTensor_base> reshape(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &Rowrank=0){
                cytnx_error_msg(true,"[ERROR] cannot reshape a UniTensor with symmetry.%s","\n");
            }
            boost::intrusive_ptr<UniTensor_base> to_dense(){
                cytnx_error_msg(true,"[ERROR] cannot to_dense a UniTensor with symmetry.%s","\n");
            }
            void to_dense_(){
                cytnx_error_msg(true,"[ERROR] cannot to_dense_ a UniTensor with symmetry.%s","\n");
            }
            void combineBonds(const std::vector<cytnx_int64> &indicators, const bool &by_label=true){};
            ~SparseUniTensor(){};
            // end virtual func
    };
    /// @endcond

    //======================================================================
    ///@brief An Enhanced tensor specifically designed for physical Tensor network simulation 
    class UniTensor{
        private:
            boost::intrusive_ptr<UniTensor_base> _impl;

        public:
            ///@cond
            UniTensor(): _impl(new UniTensor_base()){};
            UniTensor(const UniTensor &rhs){
                this->_impl = rhs._impl;
            }            
            UniTensor& operator=(const UniTensor &rhs){
                this->_impl = rhs._impl;
                return *this;
            }
            ///@endcond

            UniTensor(const Tensor &in_tensor, const cytnx_uint64 &Rowrank): _impl(new UniTensor_base()){
                this->Init(in_tensor,Rowrank);
            }


            UniTensor(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels={}, const cytnx_int64 &Rowrank=-1, const unsigned int &dtype=Type.Double, const int &device = Device.cpu, const bool &is_diag=false): _impl(new UniTensor_base()){
                this->Init(bonds,in_labels,Rowrank,dtype,device,is_diag);
            }

            void Init(const Tensor &in_tensor, const cytnx_uint64 &Rowrank){
               boost::intrusive_ptr<UniTensor_base> out(new DenseUniTensor());
               out->Init_by_Tensor(in_tensor, Rowrank);
               this->_impl = out;
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
                    cytnx_error_msg(true,"[interrupt, developing][SparseUniTensor]%s","\n");
                    boost::intrusive_ptr<UniTensor_base> out(new SparseUniTensor());
                    this->_impl = out;
                }else{   
                    boost::intrusive_ptr<UniTensor_base> out(new DenseUniTensor());
                    this->_impl = out;
                }
                this->_impl->Init(bonds, in_labels, Rowrank, dtype, device, is_diag);
            }
            
            void set_name(const std::string &in){
                this->_impl->set_name(in);
            }
            void set_label(const cytnx_uint64 &idx, const cytnx_int64 &new_label){
                this->_impl->set_label(idx,new_label);
            }
            void set_labels(const std::vector<cytnx_int64> &new_labels){
                this->_impl->set_labels(new_labels);
            }
            void set_Rowrank(const cytnx_uint64 &new_Rowrank){
                this->_impl->set_Rowrank(new_Rowrank);
            }
            template<class T>
            T& item(){

                cytnx_error_msg(this->is_blockform(),"[ERROR] cannot use item on UniTensor with Symmetry.\n suggestion: use get_block()/get_blocks() first.%s","\n");
                
                DenseUniTensor* tmp = static_cast<DenseUniTensor*>(this->_impl.get());
                return tmp->_block.item<T>();

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
            
            template<class T>
            T& at(const std::vector<cytnx_uint64> &locator){
                return this->_impl->at<T>(locator);
            }
            // return a clone of block
            Tensor get_block(const cytnx_uint64 &idx=0) const{
                return this->_impl->get_block(idx);
            };
            // return a clone of block
            Tensor get_block(const std::vector<cytnx_int64> &qnum) const{
                return this->_impl->get_block(qnum);
            }
            // this only work for non-symm tensor. return a shared view of block
            Tensor get_block_(const cytnx_uint64 &idx=0) const{
                return this->_impl->get_block_(idx);
            }
            // this return a shared view of blocks for non-symm tensor.
            // for symmetry tensor, it call contiguous first and return a shared view of blocks. [dev]
            std::vector<Tensor> get_blocks() const {
                return this->_impl->get_blocks();
            }
            // the put block will have shared view with the internal block, i.e. non-clone. 
            void put_block(const Tensor &in,const cytnx_uint64 &idx=0){
                this->_impl->put_block(in,idx);
            }
            // the put block will have shared view with the internal block, i.e. non-clone. 
            void put_block(const Tensor &in, const std::vector<cytnx_int64> &qnum){
                this->_impl->put_block(in,qnum);
            }

            UniTensor get(const std::vector<Accessor> &accessors) const{
                UniTensor out;
                out._impl = this->_impl->get(accessors);
                return out;
            }
            void set(const std::vector<Accessor> &accessors, const Tensor &rhs){
                this->_impl->set(accessors, rhs);
            }

            UniTensor reshape(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &Rowrank=0){
                UniTensor out;
                out._impl = this->_impl->reshape(new_shape,Rowrank);
                return out;
            }
            void reshape_(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &Rowrank=0){
                this->_impl->reshape_(new_shape,Rowrank);
            }
            

    };

    std::ostream& operator<<(std::ostream& os, const UniTensor &in);

}

#endif
