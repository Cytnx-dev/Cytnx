#ifndef _H_linalg_internal_interface_
#define _H_linalg_internal_interface_
#include <iostream>
#include <vector>

#include "Type.hpp"
#include "Storage.hpp"
#include "linalg/linalg_internal_cpu/Arithmic_internal.hpp"
#include "linalg/linalg_internal_cpu/Svd_internal.hpp"
#include "linalg/linalg_internal_cpu/Inv_inplace_internal.hpp"
#include "linalg/linalg_internal_cpu/Conj_inplace_internal.hpp"
#ifdef UNI_GPU
    #include "linalg/linalg_internal_gpu/cuArithmic_internal.hpp"
    #include "linalg/linalg_internal_gpu/cuSvd_internal.hpp"
    #include "linalg/linalg_internal_gpu/cuInv_inplace_internal.hpp"
    #include "linalg/linalg_internal_gpu/cuConj_inplace_internal.hpp"
#endif

namespace cytnx{
    namespace linalg_internal{
        typedef void (*Arithmicfunc_oii)(boost::intrusive_ptr<Storage_base> &,boost::intrusive_ptr<Storage_base> &,boost::intrusive_ptr<Storage_base> &,const unsigned long long & len, const char &type);
        typedef void (*Svdfunc_oii)(const boost::intrusive_ptr<Storage_base> &, boost::intrusive_ptr<Storage_base> &,boost::intrusive_ptr<Storage_base> &,  boost::intrusive_ptr<Storage_base> &, const cytnx_int32 &, const cytnx_int32&);
        typedef void (*Invinplacefunc_oii)(boost::intrusive_ptr<Storage_base> &, const cytnx_int32&);
        typedef void (*Conjinplacefunc_oii)(boost::intrusive_ptr<Storage_base> &, const cytnx_uint64 &);

        class linalg_internal_interface{
            public:
                std::vector<std::vector<Arithmicfunc_oii> > Ari_ii;
                std::vector<Svdfunc_oii> Svd_ii;
                std::vector<Invinplacefunc_oii> Inv_inplace_ii;
                std::vector<Conjinplacefunc_oii> Conj_inplace_ii;

                #ifdef UNI_GPU
                std::vector<std::vector<Arithmicfunc_oii> > cuAri_ii;
                std::vector<Svdfunc_oii> cuSvd_ii;
                std::vector<Invinplacefunc_oii> cuInv_inplace_ii;
                std::vector<Conjinplacefunc_oii> cuConj_inplace_ii;
                #endif

            linalg_internal_interface();
                

        };
        extern linalg_internal_interface lii;
    }
}
#endif

