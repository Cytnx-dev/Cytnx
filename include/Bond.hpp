#ifndef _H_Bond_
#define _H_Bond_

#include "Type.hpp"
#include "Symmetry.hpp"
#include "cytnx_error.hpp"
#include <initializer_list>
#include <vector>
#include "intrusive_ptr_base.hpp"
#include "utils/vec_clone.hpp"
namespace cytnx{

    enum bondType: int{
        BD_KET = -1,
        BD_BRA = 1,
        BD_REG =0
    };
    /// @cond
    class Bond_impl: public intrusive_ptr_base<Bond_impl>{
        private:
            cytnx_uint64 _dim;
            bondType _type;
            std::vector< std::vector<cytnx_int64> > _qnums;
            std::vector<Symmetry> _syms;

        public:

            Bond_impl(): _type(bondType::BD_REG) {};   

            void Init(const cytnx_uint64 &dim, const bondType &bd_type=bondType::BD_REG, const std::vector<std::vector<cytnx_int64> > &in_qnums = {}, const std::vector<Symmetry> &in_syms={});


            bondType                                type() const{return this->_type;};
            const std::vector<std::vector<cytnx_int64> >& qnums() const{return this->_qnums;}
            const cytnx_uint64&                             dim() const{return this->_dim;}
            cytnx_uint32                           Nsym() const{return this->_syms.size();}
            std::vector<Symmetry>                   syms() const{return vec_clone(this->_syms);}


            void set_type(const bondType &new_bondType){
                this->_type = new_bondType;
            }

            void clear_type(){
                this->_type = bondType::BD_REG;
            }

            
            boost::intrusive_ptr<Bond_impl> clone(){
                boost::intrusive_ptr<Bond_impl> out(new Bond_impl());
                out->_dim = this->dim();
                out->_type = this->type();
                out->_qnums = this->qnums();
                out->_syms  = this->syms();// return a clone of vec!
                return out;
            }

            void combineBond_(const boost::intrusive_ptr<Bond_impl> &bd_in){
                //check:
                cytnx_error_msg(this->type() != bd_in->type(),"%s\n","[ERROR] cannot combine two Bonds with different types.");
                cytnx_error_msg(this->Nsym() != bd_in->Nsym(),"%s\n","[ERROR] cannot combine two Bonds with differnet symmetry.");
                if(this->Nsym() != 0)
                    cytnx_error_msg(this->syms() != bd_in->syms(),"%s\n","[ERROR] cannot combine two Bonds with differnet symmetry.");

                this->_dim *= bd_in->dim();
            
                /// handle symmetry
                std::vector<std::vector<cytnx_int64> > new_qnums(this->
                for(cytnx_uint32 i=0;i<this->Nsym();i++){
                
                }                        
                

            }                    

    };//Bond_impl
    ///@endcond

    //wrapper:
    class Bond{
        public:
            boost::intrusive_ptr<Bond_impl> _impl;
            Bond(): _impl(new Bond_impl()){};
            Bond(const cytnx_uint64 &dim, const bondType &bd_type=bondType::BD_REG, const std::vector<std::vector<cytnx_int64> > &in_qnums={}, const std::vector<Symmetry> &in_syms={}): _impl(new Bond_impl()){
                this->_impl->Init(dim,bd_type,in_qnums,in_syms);
            };

            Bond(const Bond&rhs){this->_impl = rhs._impl;}
            Bond& operator=(const Bond &rhs){this->_impl = rhs._impl; return *this;}

            bondType                                type() const{return this->_impl->type();};
            std::vector<std::vector<cytnx_int64> > qnums() const{return this->_impl->qnums();};
            cytnx_uint64                             dim() const{return this->_impl->dim();};
            cytnx_uint32                            Nsym() const{return this->_impl->syms().size();};
            std::vector<Symmetry>                   syms() const{return this->_impl->syms();};


            void set_type(const bondType &new_bondType){
                this->_impl->set_type(new_bondType);
            }

            void clear_type(){
                this->_impl->clear_type();
            }

            Bond clone() const{
                Bond out;
                out._impl = this->_impl->clone();
                return out;
            }

           

    };

    ///@cond
    std::ostream& operator<<(std::ostream &os,const Bond &bin);
    ///@endcond
}



#endif
