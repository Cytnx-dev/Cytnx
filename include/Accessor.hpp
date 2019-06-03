#ifndef __Accessor_H_
#define __Accessor_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include <vector>
#include <cstring>
#include <string>
namespace cytnx{
    /**
    * @brief object that mimic the python slice to access elements in C++ [this is for c++ API only]. 
    */
    class Accessor{
        private:
            cytnx_int64 type; 
            cytnx_int64 min, max, step;
            cytnx_int64 loc;

            // if type is singl, min/max/step     are not used
            // if type is all  , min/max/step/loc are not used
            // if type is range, loc              are not used.

        public:
            enum : cytnx_int64{
                none,
                Singl,
                All,
                Range
            };

            Accessor(): type(Accessor::none){};

            // singul constr.
            Accessor(const cytnx_int64 &loc);

            // all constr. ( use string to dispatch )            
            Accessor(const std::string &str);

            // range constr. 
            Accessor(const cytnx_int64 &min, const cytnx_int64 &max, const cytnx_int64 &step);


            //copy constructor:
            Accessor(const Accessor& rhs);

            //copy assignment:
            Accessor& operator=(const Accessor& rhs);

            //handy generator function :
            static Accessor all(){
                return Accessor(std::string("s"));
            };

            static Accessor range(const cytnx_int64 &min, 
                                  const cytnx_int64 &max, 
                                  const cytnx_int64 &step=1){
                return Accessor(min,max,step);
            };

            // get the real len from dim
            // if type is all, pos will be null, and len == dim
            // if type is range, pos will be the locator, and len == len(pos)
            // if type is singl, pos will be pos, and len == 0 
            void get_len_pos(const cytnx_uint64 &dim, cytnx_uint64 &len, std::vector<cytnx_uint64> &pos) const;
    };

}// namespace cytnx

#endif
