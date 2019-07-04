#ifndef _H_Network_
#define _H_Network_

#include "Type.hpp"
#include "Symmetry.hpp"
#include "cytnx_error.hpp"
#include <initializer_list>
#include <vector>
#include <map>
#include "intrusive_ptr_base.hpp"
#include "utils/vec_clone.hpp"

namespace cytnx{

    /// this 


    /// @cond
    class Network_impl: public intrusive_ptr_base<Network_impl>{
        private:

            std::string fname;
            bool load, isChanged;
            std::vector<UniTensor> tensors; // tensor_type?
            bool ordered;
           
            //Contraction order.
            std::vector<cytnx_int64> contract_order;

            // labels corr to the tn list.
            std::vector< std::vector<cytnx_int64> > label_arr;

            std::vector< cytnx_int64 > iBondNums;

            // name of tn.
            std::vector< std::string > names;
            std::map<std::string, std::vector<cytnx_int64> > name2pos;

            // [Future] Swap gates. 
            

        public:
            Network_impl(){};
            void Init(const std::string& fname);
            void print_network() const;
            
            void PreConstruct(bool force = true);
            
            void PutTensor(cytnx_int64 idx, const UniTensor& UniT, bool force = true);
            
            void PutTensor(const std::string  &name, const UniTensor &UniT, bool force = true);
            
            UniTensor Launch(const std::string &Tname="");
            
            std::string GetContractOrder() const;

                



    };//Network_impl
    ///@endcond

    /// @brief the Network
    class Network{
        public:
            ///@cond
            boost::intrusive_ptr<Network_impl> _impl;
            Network(): _impl(new Network_impl()){};
            Network(const Network&rhs){this->_impl = rhs._impl;}
            Network& operator=(const Network &rhs){this->_impl = rhs._impl; return *this;}
            ///@endcond

            Network(const cytnx_uint64 &dim, const bondType &bd_type=bondType::BD_REG, const std::vector<std::vector<cytnx_int64> > &in_qnums={}, const std::vector<Symmetry> &in_syms={}): _impl(new Network_impl()){
                this->_impl->Init(dim,bd_type,in_qnums,in_syms);
            }
            
            void Init(const cytnx_uint64 &dim, const bondType &bd_type=bondType::BD_REG, const std::vector<std::vector<cytnx_int64> > &in_qnums={}, const std::vector<Symmetry> &in_syms={}){
                this->_impl->Init(dim,bd_type,in_qnums,in_syms);
            }
            

    
    };

    ///@cond
    std::ostream& operator<<(std::ostream &os,const Network &bin);
    ///@endcond
}



#endif
