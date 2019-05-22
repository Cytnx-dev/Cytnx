#ifndef _H_Movemem_cpu_
#define _H_Movemem_cpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "../../Type.hpp"
#include "../../Storage.hpp"
#include "../../tor10_error.hpp"

namespace tor10{
    namespace utils_internal{
        
        boost::intrusive_ptr<Storage_base> Movemem_cpu_cd(boost::intrusive_ptr<Storage_base> &in, const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64>&mapper, const std::vector<tor10_uint64> &invmapper, const bool is_inplace);

        boost::intrusive_ptr<Storage_base> Movemem_cpu_cf(boost::intrusive_ptr<Storage_base> &in, const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64>&mapper, const std::vector<tor10_uint64> &invmapper, const bool is_inplace);
        
        boost::intrusive_ptr<Storage_base> Movemem_cpu_d(boost::intrusive_ptr<Storage_base> &in, const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64>&mapper, const std::vector<tor10_uint64> &invmapper, const bool is_inplace);
        
        boost::intrusive_ptr<Storage_base> Movemem_cpu_f(boost::intrusive_ptr<Storage_base> &in, const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64>&mapper, const std::vector<tor10_uint64> &invmapper, const bool is_inplace);

        boost::intrusive_ptr<Storage_base> Movemem_cpu_i64(boost::intrusive_ptr<Storage_base> &in, const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64>&mapper, const std::vector<tor10_uint64> &invmapper, const bool is_inplace);

        boost::intrusive_ptr<Storage_base> Movemem_cpu_u64(boost::intrusive_ptr<Storage_base> &in, const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64>&mapper, const std::vector<tor10_uint64> &invmapper, const bool is_inplace);

        boost::intrusive_ptr<Storage_base> Movemem_cpu_i32(boost::intrusive_ptr<Storage_base> &in, const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64>&mapper, const std::vector<tor10_uint64> &invmapper, const bool is_inplace);

        boost::intrusive_ptr<Storage_base> Movemem_cpu_u32(boost::intrusive_ptr<Storage_base> &in, const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64>&mapper, const std::vector<tor10_uint64> &invmapper, const bool is_inplace);


    }//namespace utils_internal
}//namespace tor10
#endif
