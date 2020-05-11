#include "cuVectordot_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "lapack_wrapper.hpp"

namespace cytnx{



    namespace linalg_internal{
        using namespace std;
        void cuVectordot_internal_cd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const bool &is_conj){

            cublasHandle_t cublasH = NULL;
            checkCudaErrors(cublasCreate(&cublasH));

            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;
            
            _out[0] = make_cuDoubleComplex(0.,0.);
            unsigned long long remain = len;
            unsigned long long bias = 0; 
            unsigned int TotSeg = (len/INT_MAX)+1;
            int cnt = 0;
            cytnx_int32 ONE = 1;
            cytnx_int32 MAXX = INT_MAX; 
            cuDoubleComplex *dacres;
            cudaMallocManaged((void**)&dacres,sizeof(cuDoubleComplex)*TotSeg);
            cudaMemset(dacres,0,sizeof(cuDoubleComplex)*TotSeg);

            while(remain!=0){
                cout << "cnt"<< endl;
                if(remain>=INT_MAX) MAXX = INT_MAX;
                else MAXX = remain;
                
                if(is_conj)
                    checkCudaErrors(cublasZdotc(cublasH,MAXX,&_Lin[bias],ONE,&_Rin[bias],ONE,dacres+cnt));
                else
                    checkCudaErrors(cublasZdotu(cublasH,MAXX,&_Lin[bias],ONE,&_Rin[bias],ONE,dacres+cnt));
                
                remain -= MAXX;
                bias += MAXX;
                cnt += 1;
            }

            cytnx_complex128 *hacres = (cytnx_complex128*)malloc(sizeof(cytnx_complex128)*TotSeg);
            cudaMemcpy((cuDoubleComplex*)hacres,dacres,sizeof(cytnx_complex128)*TotSeg,cudaMemcpyDeviceToHost);
            for(int i=1;i<TotSeg;i++){
                hacres[0] += hacres[i];
            }
            _out[0] = make_cuDoubleComplex(hacres[0].real(),hacres[0].imag());


            free(hacres);
            cudaFree(dacres);
            cublasDestroy(cublasH);

        }
        void cuVectordot_internal_cf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const bool &is_conj){
            cublasHandle_t cublasH = NULL;
            checkCudaErrors(cublasCreate(&cublasH));

            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;
            
            _out[0] = make_cuFloatComplex(0.,0.);
            unsigned long long remain = len;
            unsigned long long bias = 0; 
            unsigned int TotSeg = (len/INT_MAX)+1;
            int cnt = 0;
            cytnx_int32 ONE = 1;
            cytnx_int32 MAXX = INT_MAX; 
            cuFloatComplex *dacres;
            cudaMallocManaged((void**)&dacres,sizeof(cuFloatComplex)*TotSeg);
            cudaMemset(dacres,0,sizeof(cuFloatComplex)*TotSeg);

            while(remain!=0){
                if(remain>=INT_MAX) MAXX = INT_MAX;
                else MAXX = remain;
                
                if(is_conj)
                    checkCudaErrors(cublasCdotc(cublasH,MAXX,&_Lin[bias],ONE,&_Rin[bias],ONE,dacres+cnt));
                else
                    checkCudaErrors(cublasCdotu(cublasH,MAXX,&_Lin[bias],ONE,&_Rin[bias],ONE,dacres+cnt));
                
                remain -= MAXX;
                bias += MAXX;
                cnt += 1;
            }

            cytnx_complex64 *hacres = (cytnx_complex64*)malloc(sizeof(cytnx_complex64)*TotSeg);
            cudaMemcpy((cuFloatComplex*)hacres,dacres,sizeof(cytnx_complex64)*TotSeg,cudaMemcpyDeviceToHost);
            for(int i=1;i<TotSeg;i++){
                hacres[0] += hacres[i];
            }
            _out[0] = make_cuFloatComplex(hacres[0].real(),hacres[0].imag());


            free(hacres);
            cudaFree(dacres);
            cublasDestroy(cublasH);


        }
        void cuVectordot_internal_d(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const bool &is_conj){
            cublasHandle_t cublasH = NULL;
            checkCudaErrors(cublasCreate(&cublasH));

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            _out[0] = 0;
            unsigned long long remain = len;
            unsigned long long bias = 0; 
            cytnx_int32 ONE = 1;
            cytnx_int32 MAXX = INT_MAX; 
            cytnx_double *acres;
            cudaMalloc((void**)&acres,sizeof(cytnx_double));

            while(remain!=0){
                if(remain>=INT_MAX) MAXX = INT_MAX;
                else MAXX = remain;
                
                checkCudaErrors(cublasDdot(cublasH,MAXX,&_Lin[bias],ONE,&_Rin[bias],ONE,acres));
                
                _out[0] += acres[0];
                remain -= MAXX;
                bias += MAXX;
            }
            cudaFree(acres);
            cublasDestroy(cublasH);

        }
        void cuVectordot_internal_f(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const bool &is_conj){
            cublasHandle_t cublasH = NULL;
            checkCudaErrors(cublasCreate(&cublasH));

            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            _out[0] = 0;
            unsigned long long remain = len;
            unsigned long long bias = 0; 
            cytnx_int32 ONE = 1;
            cytnx_int32 MAXX = INT_MAX; 
            cytnx_float *acres;
            cudaMalloc((void**)&acres,sizeof(cytnx_float));

            while(remain!=0){
                if(remain>=INT_MAX) MAXX = INT_MAX;
                else MAXX = remain;
                
                checkCudaErrors(cublasSdot(cublasH,MAXX,&_Lin[bias],ONE,&_Rin[bias],ONE,acres));
                
                _out[0] += acres[0];
                remain -= MAXX;
                bias += MAXX;
            }
            cudaFree(acres);
            cublasDestroy(cublasH);
        }
        void cuVectordot_internal_i64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const bool &is_conj){
            cytnx_error_msg(1,"[ERROR][cuVectordot_internal_i64][FATAL Invalid internal call.] No internal function for vectordot of int64 type.%s","\n");
        }
        void cuVectordot_internal_u64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const bool &is_conj){
            cytnx_error_msg(1,"[ERROR][cuVectordot_internal_u64][FATAL Invalid internal call.] No internal function for vectordot of uint64 type.%s","\n");
        }
        void cuVectordot_internal_i32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const bool &is_conj){
            cytnx_error_msg(1,"[ERROR][cuVectordot_internal_i32][FATAL Invalid internal call.] No internal function for vectordot of int32 type.%s","\n");
        }
        void cuVectordot_internal_u32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const bool &is_conj){
            cytnx_error_msg(1,"[ERROR][cuVectordot_internal_u32][FATAL Invalid internal call.] No internal function for vectordot of uint32 type.%s","\n");
        }
        void cuVectordot_internal_i16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const bool &is_conj){
            cytnx_error_msg(1,"[ERROR][cuVectordot_internal_i16][FATAL Invalid internal call.] No internal function for vectordot of int16 type.%s","\n");
        }
        void cuVectordot_internal_u16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const bool &is_conj){
            cytnx_error_msg(1,"[ERROR][cuVectordot_internal_u16][FATAL Invalid internal call.] No internal function for vectordot of uint16 type.%s","\n");
        }
        void cuVectordot_internal_b(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const bool &is_conj){
            cytnx_error_msg(1,"[ERROR][cuVectordot_internal_b][FATAL Invalid internal call.] No internal function for vectordot of bool type.%s","\n");
        }

    }//namespace linalg_internal
}//namespace cytnx



