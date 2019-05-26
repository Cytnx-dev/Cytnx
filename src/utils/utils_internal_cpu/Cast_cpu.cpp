#include "utils/utils_internal_cpu/Cast_cpu.hpp"
#include "Storage.hpp"
#ifdef UNI_OMP
#include <omp.h>
#endif

using namespace std;

namespace cytnx{
    namespace utils_internal{

        Cast_cpu_interface::Cast_cpu_interface(){
            UElemCast_cpu = vector<vector<ElemCast_io> >(N_Type,vector<ElemCast_io>(N_Type,NULL));

            UElemCast_cpu[cytnxtype.ComplexDouble][cytnxtype.ComplexDouble] = Cast_cpu_cdtcd;
            UElemCast_cpu[cytnxtype.ComplexDouble][cytnxtype.ComplexFloat ] = Cast_cpu_cdtcf;
            //UElemCast_cpu[cytnxtype.ComplexDouble][cytnxtype.Double       ] = Cast_cpu_invalid;
            //UElemCast_cpu[cytnxtype.ComplexDouble][cytnxtype.Float        ] = Cast_cpu_invalid;
            //UElemCast_cpu[cytnxtype.ComplexDouble][cytnxtype.Int64        ] = Cast_cpu_invalid;
            //UElemCast_cpu[cytnxtype.ComplexDouble][cytnxtype.Uint64       ] = Cast_cpu_invalid;
            //UElemCast_cpu[cytnxtype.ComplexDouble][cytnxtype.Int32        ] = Cast_cpu_invalid;
            //UElemCast_cpu[cytnxtype.ComplexDouble][cytnxtype.Uint32       ] = Cast_cpu_invalid;

            UElemCast_cpu[cytnxtype.ComplexFloat][cytnxtype.ComplexDouble] = Cast_cpu_cftcd;
            UElemCast_cpu[cytnxtype.ComplexFloat][cytnxtype.ComplexFloat ] = Cast_cpu_cftcf;
            //UElemCast_cpu[cytnxtype.ComplexFloat][cytnxtype.Double       ] = Cast_cpu_invalid;
            //UElemCast_cpu[cytnxtype.ComplexFloat][cytnxtype.Float        ] = Cast_cpu_invalid;
            //UElemCast_cpu[cytnxtype.ComplexFloat][cytnxtype.Int64        ] = Cast_cpu_invalid;
            //UElemCast_cpu[cytnxtype.ComplexFloat][cytnxtype.Uint64       ] = Cast_cpu_invalid;
            //UElemCast_cpu[cytnxtype.ComplexFloat][cytnxtype.Int32        ] = Cast_cpu_invalid;
            //UElemCast_cpu[cytnxtype.ComplexFloat][cytnxtype.Uint32       ] = Cast_cpu_invalid;

            UElemCast_cpu[cytnxtype.Double][cytnxtype.ComplexDouble] = Cast_cpu_dtcd;
            UElemCast_cpu[cytnxtype.Double][cytnxtype.ComplexFloat ] = Cast_cpu_dtcf;
            UElemCast_cpu[cytnxtype.Double][cytnxtype.Double       ] = Cast_cpu_dtd;
            UElemCast_cpu[cytnxtype.Double][cytnxtype.Float        ] = Cast_cpu_dtf;
            UElemCast_cpu[cytnxtype.Double][cytnxtype.Int64        ] = Cast_cpu_dti64;
            UElemCast_cpu[cytnxtype.Double][cytnxtype.Uint64       ] = Cast_cpu_dtu64;
            UElemCast_cpu[cytnxtype.Double][cytnxtype.Int32        ] = Cast_cpu_dti32;
            UElemCast_cpu[cytnxtype.Double][cytnxtype.Uint32       ] = Cast_cpu_dtu32;

            UElemCast_cpu[cytnxtype.Float][cytnxtype.ComplexDouble] = Cast_cpu_ftcd;
            UElemCast_cpu[cytnxtype.Float][cytnxtype.ComplexFloat ] = Cast_cpu_ftcf;
            UElemCast_cpu[cytnxtype.Float][cytnxtype.Double       ] = Cast_cpu_ftd;
            UElemCast_cpu[cytnxtype.Float][cytnxtype.Float        ] = Cast_cpu_ftf;
            UElemCast_cpu[cytnxtype.Float][cytnxtype.Int64        ] = Cast_cpu_fti64;
            UElemCast_cpu[cytnxtype.Float][cytnxtype.Uint64       ] = Cast_cpu_ftu64;
            UElemCast_cpu[cytnxtype.Float][cytnxtype.Int32        ] = Cast_cpu_fti32;
            UElemCast_cpu[cytnxtype.Float][cytnxtype.Uint32       ] = Cast_cpu_ftu32;

            UElemCast_cpu[cytnxtype.Int64][cytnxtype.ComplexDouble] = Cast_cpu_i64tcd;
            UElemCast_cpu[cytnxtype.Int64][cytnxtype.ComplexFloat ] = Cast_cpu_i64tcf;
            UElemCast_cpu[cytnxtype.Int64][cytnxtype.Double       ] = Cast_cpu_i64td;
            UElemCast_cpu[cytnxtype.Int64][cytnxtype.Float        ] = Cast_cpu_i64tf;
            UElemCast_cpu[cytnxtype.Int64][cytnxtype.Int64        ] = Cast_cpu_i64ti64;
            UElemCast_cpu[cytnxtype.Int64][cytnxtype.Uint64       ] = Cast_cpu_i64tu64;
            UElemCast_cpu[cytnxtype.Int64][cytnxtype.Int32        ] = Cast_cpu_i64ti32;
            UElemCast_cpu[cytnxtype.Int64][cytnxtype.Uint32       ] = Cast_cpu_i64tu32;

            UElemCast_cpu[cytnxtype.Uint64][cytnxtype.ComplexDouble] = Cast_cpu_u64tcd;
            UElemCast_cpu[cytnxtype.Uint64][cytnxtype.ComplexFloat ] = Cast_cpu_u64tcf;
            UElemCast_cpu[cytnxtype.Uint64][cytnxtype.Double       ] = Cast_cpu_u64td;
        }
        utils_internal::Cast_cpu_interface Cast_cpu; // interface object. 


        //========================================================================
        void Cast_cpu_cdtcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
                out->Init(len_in);
            }
            memcpy(out->Mem,in->Mem,sizeof(cytnx_complex128)*len_in); 
        }

        void Cast_cpu_cdtcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
                out->Init(len_in);
            }

            cytnx_complex128* _in = static_cast<cytnx_complex128*>(in->Mem);
            cytnx_complex64*  _out= static_cast<cytnx_complex64*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }

        }

        void Cast_cpu_cftcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
                out->Init(len_in);
            }
            cytnx_complex64* _in = static_cast<cytnx_complex64*>(in->Mem);
            cytnx_complex128*  _out= static_cast<cytnx_complex128*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }

        void Cast_cpu_cftcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
                out->Init(len_in);
            }
            memcpy(out->Mem,in->Mem,sizeof(cytnx_complex64)*len_in); 

        }


        void Cast_cpu_dtcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){

            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
                out->Init(len_in);
            }
            cytnx_double* _in = static_cast<cytnx_double*>(in->Mem);
            cytnx_complex128*  _out= static_cast<cytnx_complex128*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i].real(_in[i]);
            }

        }

        void Cast_cpu_dtcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());    
                out->Init(len_in);
            }
            cytnx_double* _in = static_cast<cytnx_double*>(in->Mem);
            cytnx_complex64*  _out= static_cast<cytnx_complex64*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i].real(_in[i]);
            }
        }

        void Cast_cpu_dtd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){       
                out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
                out->Init(len_in);
            }
            memcpy(out->Mem,in->Mem,sizeof(cytnx_double)*len_in);

        }
        void Cast_cpu_dtf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
                out -> Init(len_in);
            }
            cytnx_double* _in = static_cast<cytnx_double*>(in->Mem);
            cytnx_float*  _out= static_cast<cytnx_float*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_dti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
                out-> Init(len_in);
            }
            cytnx_double* _in = static_cast<cytnx_double*>(in->Mem);
            cytnx_int64*  _out= static_cast<cytnx_int64*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_dtu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
                out->Init(len_in);
            }
            cytnx_double* _in = static_cast<cytnx_double*>(in->Mem);
            cytnx_uint64*  _out= static_cast<cytnx_uint64*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }

        }
        void Cast_cpu_dti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
                out->Init(len_in);

            }
            cytnx_double* _in = static_cast<cytnx_double*>(in->Mem);
            cytnx_int32*  _out= static_cast<cytnx_int32*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }

        }
        void Cast_cpu_dtu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
                out->Init(len_in);
            }
            cytnx_double* _in = static_cast<cytnx_double*>(in->Mem);
            cytnx_uint32*  _out= static_cast<cytnx_uint32*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }

        }

        void Cast_cpu_ftcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
                out->Init(len_in);
            }
            cytnx_float* _in = static_cast<cytnx_float*>(in->Mem);
            cytnx_complex128*  _out= static_cast<cytnx_complex128*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i].real(_in[i]);
            }
        }
        void Cast_cpu_ftcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
                out->Init(len_in);
            }
            cytnx_float* _in = static_cast<cytnx_float*>(in->Mem);
            cytnx_complex64*  _out= static_cast<cytnx_complex64*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i].real(_in[i]);
            }
        }
        void Cast_cpu_ftd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
                out->Init(len_in);
            }
            cytnx_float* _in = static_cast<cytnx_float*>(in->Mem);
            cytnx_double*  _out= static_cast<cytnx_double*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_ftf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
                out->Init(len_in);
            }
            memcpy(out->Mem,in->Mem,sizeof(cytnx_float)*len_in);
        }
        void Cast_cpu_fti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
                out->Init(len_in);
            }
            cytnx_float* _in = static_cast<cytnx_float*>(in->Mem);
            cytnx_int64*  _out= static_cast<cytnx_int64*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_ftu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
                out->Init(len_in);
            }
            cytnx_float* _in = static_cast<cytnx_float*>(in->Mem);
            cytnx_uint64*  _out= static_cast<cytnx_uint64*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_fti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
                out->Init(len_in);
            }
            cytnx_float* _in = static_cast<cytnx_float*>(in->Mem);
            cytnx_int32*  _out= static_cast<cytnx_int32*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_ftu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
                out->Init(len_in);
            }
            cytnx_float* _in = static_cast<cytnx_float*>(in->Mem);
            cytnx_uint32*  _out= static_cast<cytnx_uint32*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }

        void Cast_cpu_i64tcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
                out->Init(len_in);
            }
            cytnx_int64* _in = static_cast<cytnx_int64*>(in->Mem);
            cytnx_complex128*  _out= static_cast<cytnx_complex128*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i].real(_in[i]);
            }

        }
        void Cast_cpu_i64tcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
                out->Init(len_in);
            }
            cytnx_int64* _in = static_cast<cytnx_int64*>(in->Mem);
            cytnx_complex64*  _out= static_cast<cytnx_complex64*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i].real(_in[i]);
            }

        }
        void Cast_cpu_i64td(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
                out->Init(len_in);
            }
            cytnx_int64* _in = static_cast<cytnx_int64*>(in->Mem);
            cytnx_double*  _out= static_cast<cytnx_double*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }

        }
        void Cast_cpu_i64tf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
                out->Init(len_in);
            }
            cytnx_int64* _in = static_cast<cytnx_int64*>(in->Mem);
            cytnx_float*  _out= static_cast<cytnx_float*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }

        }
        void Cast_cpu_i64ti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
                out->Init(len_in);
            }
            memcpy(out->Mem,in->Mem,sizeof(cytnx_int64)*len_in);

        }
        void Cast_cpu_i64tu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
                out->Init(len_in);
            }
            cytnx_int64* _in = static_cast<cytnx_int64*>(in->Mem);
            cytnx_uint64*  _out= static_cast<cytnx_uint64*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_i64ti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
                out->Init(len_in);
            }
            cytnx_int64* _in = static_cast<cytnx_int64*>(in->Mem);
            cytnx_int32*  _out= static_cast<cytnx_int32*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_i64tu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
                out->Init(len_in);
            }
            cytnx_int64* _in = static_cast<cytnx_int64*>(in->Mem);
            cytnx_uint32*  _out= static_cast<cytnx_uint32*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }

        void Cast_cpu_u64tcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
                out->Init(len_in);
            }
            cytnx_uint64* _in = static_cast<cytnx_uint64*>(in->Mem);
            cytnx_complex128*  _out= static_cast<cytnx_complex128*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i].real( _in[i]);
            }
        }
        void Cast_cpu_u64tcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
                out->Init(len_in);
            }
            cytnx_uint64* _in = static_cast<cytnx_uint64*>(in->Mem);
            cytnx_complex64*  _out= static_cast<cytnx_complex64*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i].real( _in[i]);
            }
        }
        void Cast_cpu_u64td(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
                out->Init(len_in);
            }
            cytnx_uint64* _in = static_cast<cytnx_uint64*>(in->Mem);
            cytnx_double*  _out= static_cast<cytnx_double*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_u64tf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
                out->Init(len_in);
            }
            cytnx_uint64* _in = static_cast<cytnx_uint64*>(in->Mem);
            cytnx_float*  _out= static_cast<cytnx_float*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_u64ti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
                out->Init(len_in);
            }
            cytnx_uint64* _in = static_cast<cytnx_uint64*>(in->Mem);
            cytnx_int64*  _out= static_cast<cytnx_int64*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_u64tu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
                out->Init(len_in);
            }
            memcpy(out->Mem,in->Mem,sizeof(cytnx_uint64)*len_in);
           
        }
        void Cast_cpu_u64ti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
                out->Init(len_in);
            }
            cytnx_uint64* _in = static_cast<cytnx_uint64*>(in->Mem);
            cytnx_int32*  _out= static_cast<cytnx_int32*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_u64tu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
                out->Init(len_in);
            }
            cytnx_uint64* _in = static_cast<cytnx_uint64*>(in->Mem);
            cytnx_uint32*  _out= static_cast<cytnx_uint32*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }

        void Cast_cpu_i32tcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
                out->Init(len_in);
            }
            cytnx_int32* _in = static_cast<cytnx_int32*>(in->Mem);
            cytnx_complex128*  _out= static_cast<cytnx_complex128*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i].real( _in[i]);
            }
        }
        void Cast_cpu_i32tcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
                out->Init(len_in);
            }
            cytnx_int32* _in = static_cast<cytnx_int32*>(in->Mem);
            cytnx_complex64*  _out= static_cast<cytnx_complex64*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i].real( _in[i]);
            }
        }
        void Cast_cpu_i32td(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
                out->Init(len_in);
            }
            cytnx_int32* _in = static_cast<cytnx_int32*>(in->Mem);
            cytnx_double*  _out= static_cast<cytnx_double*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_i32tf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
                out->Init(len_in);
            }
            cytnx_int32* _in = static_cast<cytnx_int32*>(in->Mem);
            cytnx_float*  _out= static_cast<cytnx_float*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_i32ti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
                out->Init(len_in);
            }
            cytnx_int32* _in = static_cast<cytnx_int32*>(in->Mem);
            cytnx_int64*  _out= static_cast<cytnx_int64*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_i32tu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
                out->Init(len_in);
            }
            cytnx_int32* _in = static_cast<cytnx_int32*>(in->Mem);
            cytnx_uint64*  _out= static_cast<cytnx_uint64*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_i32ti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
                out->Init(len_in);
            }
            memcpy(out->Mem,in->Mem,sizeof(cytnx_int32)*len_in);
        }
        void Cast_cpu_i32tu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
                out->Init(len_in);
            }
            cytnx_int32* _in = static_cast<cytnx_int32*>(in->Mem);
            cytnx_uint32*  _out= static_cast<cytnx_uint32*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }

        void Cast_cpu_u32tcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
                out->Init(len_in);
            }
            cytnx_uint32* _in = static_cast<cytnx_uint32*>(in->Mem);
            cytnx_complex128*  _out= static_cast<cytnx_complex128*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i].real( _in[i]);
            }

        }
        void Cast_cpu_u32tcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
                out->Init(len_in);
            }
            cytnx_uint32* _in = static_cast<cytnx_uint32*>(in->Mem);
            cytnx_complex64*  _out= static_cast<cytnx_complex64*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i].real( _in[i]);
            }
        }
        void Cast_cpu_u32td(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
                out->Init(len_in);
            }
            cytnx_uint32* _in = static_cast<cytnx_uint32*>(in->Mem);
            cytnx_double*  _out= static_cast<cytnx_double*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_u32tf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
                out->Init(len_in);
            }
            cytnx_uint32* _in = static_cast<cytnx_uint32*>(in->Mem);
            cytnx_float*  _out= static_cast<cytnx_float*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_u32ti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
                out->Init(len_in);
            }
            cytnx_uint32* _in = static_cast<cytnx_uint32*>(in->Mem);
            cytnx_int64*  _out= static_cast<cytnx_int64*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_u32tu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
                out->Init(len_in);
            }
            cytnx_uint32* _in = static_cast<cytnx_uint32*>(in->Mem);
            cytnx_uint64*  _out= static_cast<cytnx_uint64*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_u32ti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
                out->Init(len_in);
            }
            cytnx_uint32* _in = static_cast<cytnx_uint32*>(in->Mem);
            cytnx_int32*  _out= static_cast<cytnx_int32*>(out->Mem);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(unsigned long long i=0;i<len_in;i++){
                _out[i] = _in[i];
            }
        }
        void Cast_cpu_u32tu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const bool &is_alloc){
            if(is_alloc){
                out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
                out->Init(len_in);
            }
            memcpy(out->Mem,in->Mem,sizeof(cytnx_uint32)*len_in);

        }
    }//namespace utils_internal
}//namespace cytnx
