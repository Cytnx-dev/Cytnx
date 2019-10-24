#include "Tridiag_internal.hpp"
#include "cytnx_error.hpp"
#include "lapack_wrapper.hpp"
#include <iostream>
namespace cytnx{

    namespace linalg_internal{

        void Tridiag_internal_d( const boost::intrusive_ptr<Storage_base> &diag, const boost::intrusive_ptr<Storage_base> &s_diag, boost::intrusive_ptr<Storage_base> &S, boost::intrusive_ptr<Storage_base> &U, const cytnx_int32 &L){
            
            char job;
            job = (U->dtype ==Type.Void) ? 'N': 'V';
            std::cout << L << std::endl;       
            //copy from in to S[out]     
            memcpy(S->Mem,diag->Mem, L * sizeof(cytnx_double));

            //create tmp for sub-diag and cpy in:
            cytnx_double* Dsv = (cytnx_double*)malloc((L-1) * sizeof(cytnx_double));
            memcpy(Dsv,s_diag->Mem, (L-1) * sizeof(cytnx_double));

            cytnx_int32 ldz = 1;
            cytnx_int32 info = 0;
            double *work;

            //check if compute eigV 
            if(U->dtype != Type.Void){
                //cytnx_error_msg((k<1) || (k>L),"[interal][Tridiag_internal_d] error, compute eigenvalue should have k>=1 and k<=L %s","\n");
                ldz = L;
                work = (double*)malloc(sizeof(double)*(2*L-2));
            }

            dstev(&job,&L,(cytnx_double*)S->Mem, Dsv, (cytnx_double*)U->Mem,&ldz,work,&info);
            cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'dstev': Lapack INFO = ", info);

            //house keeping
            if(U->dtype != Type.Void) free(work);
            free(Dsv);

        }
        void Tridiag_internal_f( const boost::intrusive_ptr<Storage_base> &diag, const boost::intrusive_ptr<Storage_base> &s_diag, boost::intrusive_ptr<Storage_base> &S, boost::intrusive_ptr<Storage_base> &U, const cytnx_int32 &L){
        }
    }//linalg_internal 
}//cytnx



