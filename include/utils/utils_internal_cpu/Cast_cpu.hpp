#ifndef _H_Cast_cpu_
#define _H_Cast_cpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "../../Type.hpp"
#include "../../Storage.hpp"
#include "../../cytnx_error.hpp"
namespace cytnx{
    namespace utils_internal{
        
        typedef void (*ElemCast_io)(const boost::intrusive_ptr<Storage_base>&,boost::intrusive_ptr<Storage_base>&,const unsigned long long &, const bool &);

        
        class Cast_cpu_interface{
            public:
                std::vector<std::vector<ElemCast_io> > UElemCast_cpu;
                Cast_cpu_interface();
        };
        extern Cast_cpu_interface Cast_cpu;
        

        void Cast_cpu_cdtcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_cdtcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);

        void Cast_cpu_cftcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_cftcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);

        void Cast_cpu_dtcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_dtcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_dtd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_dtf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_dti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_dtu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_dti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_dtu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);

        void Cast_cpu_ftcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_ftcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_ftd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_ftf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_fti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_ftu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_fti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_ftu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);

        void Cast_cpu_i64tcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_i64tcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_i64td(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_i64tf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_i64ti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_i64tu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_i64ti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_i64tu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);

        void Cast_cpu_u64tcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_u64tcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_u64td(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_u64tf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_u64ti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_u64tu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_u64ti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_u64tu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);

        void Cast_cpu_i32tcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_i32tcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_i32td(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_i32tf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_i32ti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_i32tu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_i32ti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_i32tu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);

        void Cast_cpu_u32tcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_u32tcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_u32td(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_u32tf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_u32ti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_u32tu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_u32ti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
        void Cast_cpu_u32tu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc);
    }//namespace utils_internal

}//namespace cytnx

#endif
