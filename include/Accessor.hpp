#ifndef __Accessor_H_
#define __Accessor_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include <vector>
#include <cstring>
#include <string>
#include <iostream>
namespace cytnx{
    /**
    * @brief object that mimic the python slice to access elements in C++ [this is for c++ API only]. 
    */
    class Accessor{
        private:
            cytnx_int64 _type; 

        public:
            ///@cond
            cytnx_int64 min{}, max{}, step{};
            cytnx_int64 loc{};
            // if type is singl, min/max/step     are not used
            // if type is all  , min/max/step/loc are not used
            // if type is range, loc              are not used.


            enum : cytnx_int64{
                none,
                Singl,
                All,
                Range
            };

            Accessor(): _type(Accessor::none){};
            ///@endcond

            // single constructor
            /**
            @brief access the specific index at the assigned rank in Tensor.
            @param loc the specify index 

                See also \link cytnx::Tensor.get() cytnx::Tensor.get() \endlink for how to using them.
            
            ## Example:
            ### c++ API:
            \include example/Accessor/example.cpp
            #### output>
            \verbinclude example/Accessor/example.cpp.out
            ### python API:
            \include example/Accessor/example.py               
            #### output>
            \verbinclude example/Accessor/example.py.out
            */
            explicit Accessor(const cytnx_int64 &loc);

            ///@cond
            // all constr. ( use string to dispatch )            
            explicit Accessor(const std::string &str);

            // range constr. 
            Accessor(const cytnx_int64 &min, const cytnx_int64 &max, const cytnx_int64 &step);

            //copy constructor:
            Accessor(const Accessor& rhs);
            //copy assignment:
            Accessor& operator=(const Accessor& rhs);
            ///@endcond

            int type() const{
                return this->_type;
            }




            //handy generator function :
            /**
            @brief access the whole rank, this is similar to [:] in python 
            
            ## Example:
            ### c++ API:
            \include example/Accessor/example.cpp
            #### output>
            \verbinclude example/Accessor/example.cpp.out
            ### python API:
            \include example/Accessor/example.py               
            #### output>
            \verbinclude example/Accessor/example.py.out
            */
            static Accessor all(){
                return Accessor(std::string(":"));
            };


            /**
            @brief access the range at assigned rank, this is similar to [min:max:step] in python 
            @param min 
            @param max
            @param step
            
            ## Example:
            ### c++ API:
            \include example/Accessor/example.cpp
            #### output>
            \verbinclude example/Accessor/example.cpp.out
            ### python API:
            \include example/Accessor/example.py               
            #### output>
            \verbinclude example/Accessor/example.py.out
            */
            static Accessor range(const cytnx_int64 &min, 
                                  const cytnx_int64 &max, 
                                  const cytnx_int64 &step=1){
                return Accessor(min,max,step);
            };

            ///@cond
            // get the real len from dim
            // if type is all, pos will be null, and len == dim
            // if type is range, pos will be the locator, and len == len(pos)
            // if type is singl, pos will be pos, and len == 0 
            void get_len_pos(const cytnx_uint64 &dim, cytnx_uint64 &len, std::vector<cytnx_uint64> &pos) const;
            ///@endcond
    };//class Accessor

    ///@cond
    /// layout:
    std::ostream& operator<<(std::ostream& os, const Accessor &in);

    // elements resolver
    template<class T>
    void _resolve_elems(std::vector<cytnx::Accessor> &cool, const T& a){
        cool.push_back(cytnx::Accessor(a));
    }

    template<class T, class ... Ts>
    void _resolve_elems(std::vector<cytnx::Accessor> &cool,const T&a, const Ts&... args){
        cool.push_back(cytnx::Accessor(a));
        _resolve_elems(cool,args...);
    } 

    template<class T,class ... Ts>
    std::vector<cytnx::Accessor> Indices_resolver(const T&a, const Ts&... args){
        //std::cout << a << std::endl;;
        std::vector<cytnx::Accessor> idxs;
        _resolve_elems(idxs,a,args...);
        //cout << idxs << endl;
        return idxs;
    }



    ///@endcond

}// namespace cytnx

#endif
