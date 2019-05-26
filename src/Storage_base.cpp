#include "Storage.hpp"
#include "utils/utils_internal.hpp"

using namespace std;

namespace cytnx{

    //Storage Init interface.    
    //=============================
    boost::intrusive_ptr<Storage_base> SIInit_cd(){
        boost::intrusive_ptr<Storage_base> out(new ComplexDoubleStorage());
        return out;
    }
    boost::intrusive_ptr<Storage_base> SIInit_cf(){
        boost::intrusive_ptr<Storage_base> out(new ComplexFloatStorage());
        return out;
    }
    boost::intrusive_ptr<Storage_base> SIInit_d(){
        boost::intrusive_ptr<Storage_base> out(new DoubleStorage());
        return out;
    }
    boost::intrusive_ptr<Storage_base> SIInit_f(){
        boost::intrusive_ptr<Storage_base> out(new FloatStorage());
        return out;
    }
    boost::intrusive_ptr<Storage_base> SIInit_u64(){
        boost::intrusive_ptr<Storage_base> out(new Uint64Storage());
        return out;
    }
    boost::intrusive_ptr<Storage_base> SIInit_i64(){
        boost::intrusive_ptr<Storage_base> out(new Int64Storage());
        return out;
    }
    boost::intrusive_ptr<Storage_base> SIInit_u32(){
        boost::intrusive_ptr<Storage_base> out(new Uint32Storage());
        return out;
    }
    boost::intrusive_ptr<Storage_base> SIInit_i32(){
        boost::intrusive_ptr<Storage_base> out(new Int32Storage());
        return out;
    }

    Storage_init_interface::Storage_init_interface(){
        USIInit.resize(N_Type);
        USIInit[this->Double] = SIInit_d;
        USIInit[this->Float] = SIInit_f;
        USIInit[this->ComplexDouble] = SIInit_cd;
        USIInit[this->ComplexFloat] = SIInit_cf;
        USIInit[this->Uint64] = SIInit_u64;
        USIInit[this->Int64] = SIInit_i64;
        USIInit[this->Uint32]= SIInit_u32;
        USIInit[this->Int32] = SIInit_i32;
    }

    //==========================
    void Storage_base::Init(const unsigned long long &len_in,const int &device){
        //cout << "Base.init" << endl;
        
    }           


    Storage_base::Storage_base(const unsigned long long &len_in,const int &device){
        this->Init(len_in,device);
    }


    Storage_base& Storage_base::operator=(Storage_base &Rhs){
        cout << "dev"<< endl;
        return *this;
    }

    Storage_base::Storage_base(Storage_base &Rhs){
        cout << "dev" << endl;
    }

    boost::intrusive_ptr<Storage_base> Storage_base::astype(const unsigned int &dtype){
        boost::intrusive_ptr<Storage_base> out(new Storage_base());
        if(dtype == this->dtype) return boost::intrusive_ptr<Storage_base>(this);

        if(this->device==cytnxdevice.cpu){
            if(utils_internal::Cast_cpu.UElemCast_cpu[this->dtype][dtype]==NULL){
                cytnx_error_msg(1, "[ERROR] not support type with dtype=%d",dtype);
            }else{
                utils_internal::Cast_cpu.UElemCast_cpu[this->dtype][dtype](this,out,this->len,1);
            }
        }else{
            #ifdef UNI_GPU
                if(utils_internal::cuCast_gpu.UElemCast_gpu[this->dtype][dtype]==NULL){
                    cytnx_error_msg(1,"[ERROR] not support type with dtype=%d",dtype);
                }else{
                    utils_internal::cuCast_gpu.UElemCast_gpu[this->dtype][dtype](this,out,this->len,this->device);
                }
            #else
                cytnx_error_msg(1,"%s","[ERROR][Internal Error] enter GPU section without CUDA support @ Storage.astype()");
            #endif

        }
        return out;
    }

    boost::intrusive_ptr<Storage_base> Storage_base::_create_new_sametype(){
        cytnx_error_msg(1,"%s","[ERROR] call _create_new_sametype in base");
    }

    boost::intrusive_ptr<Storage_base> Storage_base::clone(){
        boost::intrusive_ptr<Storage_base> out(new Storage_base());
        return out;
    }

    string Storage_base::dtype_str(){
        return cytnxtype.getname(this->dtype);
    }

    void Storage_base::_Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device){
        cytnx_error_msg(1,"%s","[ERROR] call _Init_byptr in base");
    }

    Storage_base::~Storage_base(){
        //cout << "delet" << endl;
        if(Mem != NULL){
            if(this->device==cytnxdevice.cpu){
                free(Mem);
            }else{
                #ifdef UNI_GPU
                    cudaFree(Mem);
                #else
                    cytnx_error_msg(1,"%s","[ERROR] trying to free an GPU memory without CUDA install");
                #endif
            }
        }
    }


    void Storage_base::Move_memory_(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper){
        cytnx_error_msg(1,"%s","[ERROR] call Move_memory_ directly on Void Storage.");
    }

    boost::intrusive_ptr<Storage_base> Storage_base::Move_memory(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper){
        cytnx_error_msg(1,"%s","[ERROR] call Move_memory_ directly on Void Storage.");
    }

    void Storage_base::to_(const int &device){
        cytnx_error_msg(1,"%s","[ERROR] call to_ directly on Void Storage.");
    }

    boost::intrusive_ptr<Storage_base> Storage_base::to(const int &device){
        cytnx_error_msg(1,"%s","[ERROR] call to directly on Void Storage.");
    }


    void Storage_base::PrintElem_byShape(std::ostream &os, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper){
        cytnx_error_msg(1,"%s","[ERROR] call PrintElem_byShape directly on Void Storage.");
    }


    void Storage_base::print_info(){
        cout << "dtype : " << this->dtype_str() << endl;
        cout << "device: " << cytnxdevice.getname(this->device) << endl;
        cout << "size  : " << this->len << endl;
    }
    void Storage_base::print_elems(){
        cytnx_error_msg(1,"%s","[ERROR] call print_elems directly on Void Storage.");
    }
    void Storage_base::print(){
        this->print_info();
        this->print_elems();
    }


    //instantiation:
    //================================================
    template<>
    float* Storage_base::data<float>(){

        //check type 
        cytnx_error_msg(dtype != cytnxtype.Float, "[ERROR] type mismatch. try to get <float> type from raw data of type %s", cytnxtype.getname(dtype).c_str());
    #ifdef UNI_GPU
        cudaDeviceSynchronize();
    #endif
        return static_cast<float*>(this->Mem);
    }
    template<>
    double* Storage_base::data<double>(){

        cytnx_error_msg(dtype != cytnxtype.Double, "[ERROR] type mismatch. try to get <double> type from raw data of type %s", cytnxtype.getname(dtype).c_str());
    #ifdef UNI_GPU
        cudaDeviceSynchronize();
    #endif
        return static_cast<double*>(this->Mem);
    }

    template<>
    std::complex<double>* Storage_base::data<std::complex<double> >(){

        cytnx_error_msg(dtype != cytnxtype.ComplexDouble, "[ERROR] type mismatch. try to get < complex<double> > type from raw data of type %s", cytnxtype.getname(dtype).c_str());
    #ifdef UNI_GPU
        cytnx_error_msg(this->device!=cytnxdevice.cpu, "%s","[ERROR] the Storage is on GPU but try to get with CUDA complex type complex<double>. use type <cuDoubleComplex>  instead.");
        cudaDeviceSynchronize();
    #endif
        return static_cast<std::complex<double>*>(this->Mem);
    }

    template<>
    std::complex<float>* Storage_base::data<std::complex<float> >(){

        cytnx_error_msg(dtype != cytnxtype.ComplexFloat, "[ERROR] type mismatch. try to get < complex<float> > type from raw data of type %s", cytnxtype.getname(dtype).c_str());
    #ifdef UNI_GPU
        cytnx_error_msg(this->device!=cytnxdevice.cpu, "%s","[ERROR] the Storage is on GPU but try to get with CUDA complex type complex<float>. use type <cuFloatComplex>  instead.");
        cudaDeviceSynchronize();
    #endif
        return static_cast<std::complex<float>*>(this->Mem);
    }

    template<>
    uint32_t* Storage_base::data<uint32_t>(){

        cytnx_error_msg(dtype != cytnxtype.Uint32, "[ERROR] type mismatch. try to get <uint32_t> type from raw data of type %s", cytnxtype.getname(dtype).c_str());
    #ifdef UNI_GPU
        cudaDeviceSynchronize();
    #endif
        return static_cast<uint32_t*>(this->Mem);
    }

    template<>
    int32_t* Storage_base::data<int32_t>(){

        cytnx_error_msg(dtype != cytnxtype.Int32, "[ERROR] type mismatch. try to get <int32_t> type from raw data of type %s", cytnxtype.getname(dtype).c_str());
    #ifdef UNI_GPU
        cudaDeviceSynchronize();
    #endif
        return static_cast<int32_t*>(this->Mem);
    }

    template<>
    uint64_t* Storage_base::data<uint64_t>(){

        cytnx_error_msg(dtype != cytnxtype.Uint64, "[ERROR] type mismatch. try to get <uint64_t> type from raw data of type %s", cytnxtype.getname(dtype).c_str());
    #ifdef UNI_GPU
        cudaDeviceSynchronize();
    #endif
        return static_cast<uint64_t*>(this->Mem);
    }

    template<>
    int64_t* Storage_base::data<int64_t>(){

        cytnx_error_msg(dtype != cytnxtype.Int64, "[ERROR] type mismatch. try to get <int64_t> type from raw data of type %s", cytnxtype.getname(dtype).c_str());
    #ifdef UNI_GPU
        cudaDeviceSynchronize();
    #endif
        return static_cast<int64_t*>(this->Mem);
    }

    // get complex raw pointer using CUDA complex type 
    #ifdef UNI_GPU
    template<>
    cuDoubleComplex* Storage_base::data<cuDoubleComplex>(){
        cytnx_error_msg(dtype != cytnxtype.ComplexDouble, "[ERROR] type mismatch. try to get <cuDoubleComplex> type from raw data of type %s", cytnxtype.getname(dtype).c_str());
        cytnx_error_msg(this->device==cytnxdevice.cpu, "%s","[ERROR] the Storage is on CPU(Host) but try to get with CUDA complex type cuDoubleComplex. use type <cytnx_complex128> or < complex<double> > instead.");
        cudaDeviceSynchronize();
        return static_cast<cuDoubleComplex*>(this->Mem);

    }
    template<>
    cuFloatComplex* Storage_base::data<cuFloatComplex>(){
        cytnx_error_msg(dtype != cytnxtype.ComplexFloat, "[ERROR] type mismatch. try to get <cuFloatComplex> type from raw data of type %s", cytnxtype.getname(dtype).c_str());
        cytnx_error_msg(this->device==cytnxdevice.cpu, "%s","[ERROR] the Storage is on CPU(Host) but try to get with CUDA complex type cuFloatComplex. use type <cytnx_complex64> or < complex<float> > instead.");
        cudaDeviceSynchronize();
        return static_cast<cuFloatComplex*>(this->Mem);

    }
    #endif

    //instantiation:
    //====================================================
    template<>
    float& Storage_base::at<float>(const unsigned int &idx){
        cytnx_error_msg(dtype != cytnxtype.Float, "[ERROR] type mismatch. try to get <float> type from raw data of type %s", cytnxtype.getname(dtype).c_str());
    #ifdef UNI_GPU
        cudaDeviceSynchronize();
    #endif
        return static_cast<float*>(this->Mem)[idx];
    }
    template<>
    double& Storage_base::at<double>(const unsigned int &idx){
        cytnx_error_msg(dtype != cytnxtype.Double, "[ERROR] type mismatch. try to get <double> type from raw data of type %s", cytnxtype.getname(dtype).c_str());
    #ifdef UNI_GPU
        cudaDeviceSynchronize();
    #endif
        return static_cast<double*>(this->Mem)[idx];
    }

    template<>
    std::complex<float>& Storage_base::at<std::complex<float> >(const unsigned int &idx){
        cytnx_error_msg(dtype != cytnxtype.ComplexFloat, "[ERROR] type mismatch. try to get < complex<float> > type from raw data of type %s", cytnxtype.getname(dtype).c_str());
    #ifdef UNI_GPU
        cudaDeviceSynchronize();
    #endif

        return static_cast<complex<float>*>(this->Mem)[idx];
    }
    template<>
    std::complex<double>& Storage_base::at<std::complex<double> >(const unsigned int &idx){
        cytnx_error_msg(dtype != cytnxtype.ComplexDouble, "[ERROR] type mismatch. try to get < complex<double> > type from raw data of type %s", cytnxtype.getname(dtype).c_str());
    #ifdef UNI_GPU
        cudaDeviceSynchronize();
    #endif
        return static_cast<complex<double>*>(this->Mem)[idx];
    }

    template<>
    uint32_t& Storage_base::at<uint32_t>(const unsigned int &idx){
        cytnx_error_msg(dtype != cytnxtype.Uint32, "[ERROR] type mismatch. try to get <uint32_t> type from raw data of type %s", cytnxtype.getname(dtype).c_str());
    #ifdef UNI_GPU
        cudaDeviceSynchronize();
    #endif
        return static_cast<uint32_t*>(this->Mem)[idx];
    }

    template<>
    int32_t& Storage_base::at<int32_t>(const unsigned int &idx){
        cytnx_error_msg(dtype != cytnxtype.Int32, "[ERROR] type mismatch. try to get <int32_t> type from raw data of type %s", cytnxtype.getname(dtype).c_str());
    #ifdef UNI_GPU
        cudaDeviceSynchronize();
    #endif
        return static_cast<int32_t*>(this->Mem)[idx];
    }

    template<>
    uint64_t& Storage_base::at<uint64_t>(const unsigned int &idx){
        cytnx_error_msg(dtype != cytnxtype.Uint64, "[ERROR] type mismatch. try to get <uint64_t> type from raw data of type %s", cytnxtype.getname(dtype).c_str());
    #ifdef UNI_GPU
        cudaDeviceSynchronize();
    #endif
        return static_cast<uint64_t*>(this->Mem)[idx];
    }

    template<>
    int64_t& Storage_base::at<int64_t>(const unsigned int &idx){
        cytnx_error_msg(dtype != cytnxtype.Int64, "[ERROR] type mismatch. try to get <int64_t> type from raw data of type %s", cytnxtype.getname(dtype).c_str());
    #ifdef UNI_GPU
        cudaDeviceSynchronize();
    #endif
        return static_cast<int64_t*>(this->Mem)[idx];
    }

}//namespace cytnx;
