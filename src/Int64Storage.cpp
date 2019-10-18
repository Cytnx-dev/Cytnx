#include "Storage.hpp"
#include "utils/utils_internal_interface.hpp"
using namespace std;

namespace cytnx{
    void Int64Storage::Init(const unsigned long long &len_in, const int &device){
        //cout << "Int64.init" << endl;
        this->len = len_in;

        //check:
        cytnx_error_msg(len_in < 1, "%s", "[ERROR] cannot init a Storage with zero element");
        this->dtype = Type.Int64;
        if (device == Device.cpu){
            this->Mem = utils_internal::Malloc_cpu(this->len*sizeof(cytnx_int64));
        }else{
            #ifdef UNI_GPU
                cytnx_error_msg(device>=Device.Ngpus,"%s","[ERROR] invalid device.");
                cudaSetDevice(device);
                this->Mem = utils_internal::cuMalloc_gpu(this->len*sizeof(cytnx_int64));
            #else
                cytnx_error_msg(1,"%s","[ERROR] cannot init a Storage on gpu without CUDA support.");
            #endif
        }
        this->device =device;
    }


    void Int64Storage::_Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device){
        this->Mem = rawptr;
        this->len = len_in;
    # ifdef UNI_DEBUG
        cytnx_error_msg(len_in < 1, "%s", "[ERROR] _Init_by_ptr cannot have len_in < 1.");
    # endif
        this->dtype = Type.Int64;
        this->device = device;
    }

    boost::intrusive_ptr<Storage_base> Int64Storage::_create_new_sametype(){
        boost::intrusive_ptr<Storage_base> out(new Int64Storage());
        return out;
    }

    boost::intrusive_ptr<Storage_base> Int64Storage::clone(){
        boost::intrusive_ptr<Storage_base> out(new Int64Storage());
        out->Init(this->len,this->device);
        if(this->device==Device.cpu){
            memcpy(out->Mem,this->Mem,sizeof(cytnx_int64)*this->len);
        }else{
            #ifdef UNI_GPU
                checkCudaErrors(cudaSetDevice(this->device));
                checkCudaErrors(cudaMemcpy(out->Mem,this->Mem,sizeof(cytnx_int64)*this->len,cudaMemcpyDeviceToDevice));
            #else
                cytnx_error_msg(1,"%s","[ERROR] cannot clone a Storage on gpu without CUDA support.");
            #endif
        }
        return out;
    }

    void Int64Storage::Move_memory_(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper){
        boost::intrusive_ptr<Storage_base> tmp(this);
        if(this->device==Device.cpu){
            utils_internal::Movemem_cpu_i64(tmp,old_shape,mapper,invmapper,1);
        }else{
            #ifdef UNI_GPU
                checkCudaErrors(cudaSetDevice(this->device));
                utils_internal::cuMovemem_gpu_i64(tmp,old_shape,mapper,invmapper,1); 
            #else
                cytnx_error_msg(1,"%s","[ERROR][Internal] try to call GPU section without CUDA support");
            #endif
        }

    }

    boost::intrusive_ptr<Storage_base> Int64Storage::Move_memory(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper){
        boost::intrusive_ptr<Storage_base> tmp(this);
        if(this->device==Device.cpu){
            return utils_internal::Movemem_cpu_i64(tmp,old_shape,mapper,invmapper,0);
        }else{
            #ifdef UNI_GPU
                checkCudaErrors(cudaSetDevice(this->device));
                return utils_internal::cuMovemem_gpu_i64(tmp,old_shape,mapper,invmapper,0); 
            #else
                cytnx_error_msg(1,"%s","[ERROR][Internal] try to call GPU section without CUDA support");
                return nullptr;
            #endif
        }

    }

    void Int64Storage::to_(const int &device){
        if(this->device != device){
            if(this->device==Device.cpu){
                //here, cpu->gpu with gid=device
                #ifdef UNI_GPU
                    cytnx_error_msg(device>=Device.Ngpus,"%s","[ERROR] invalid device.");
                    cudaSetDevice(device);          
                    void *dtmp = utils_internal::cuMalloc_gpu(sizeof(cytnx_int64)*this->len);
                    checkCudaErrors(cudaMemcpy(dtmp,this->Mem,sizeof(cytnx_int64)*this->len,cudaMemcpyHostToDevice));
                    free(this->Mem);
                    this->Mem = dtmp;
                    this->device=device;
                #else
                    cytnx_error_msg(1,"%s","[ERROR] try to move from cpu(Host) to gpu without CUDA support."); 
                #endif
            }else{
                #ifdef UNI_GPU
                    if(device==Device.cpu){
                        //here, gpu->cpu
                        cudaSetDevice(this->device);
                        void *htmp = malloc(sizeof(cytnx_int64)*this->len);
                        checkCudaErrors(cudaMemcpy(htmp,this->Mem,sizeof(cytnx_int64)*this->len,cudaMemcpyDeviceToHost));
                        cudaFree(this->Mem);
                        this->Mem = htmp;
                        this->device = device;
                    }else{
                        // here, gpu->gpu 
                        cytnx_error_msg(device>=Device.Ngpus,"%s","[ERROR] invalid device.");
                        cudaSetDevice(device);
                        void *dtmp = utils_internal::cuMalloc_gpu(sizeof(cytnx_int64)*this->len);
                        checkCudaErrors(cudaMemcpyPeer(dtmp,device,this->Mem,this->device,sizeof(cytnx_int64)*this->len));
                        cudaFree(this->Mem);
                        this->Mem = dtmp;
                        this->device = device;
                    }
                #else
                    cytnx_error_msg(1,"%s","[ERROR][Internal] Storage.to_. the Storage is as GPU but without CUDA support.");
                #endif
            }
        }
    }
    boost::intrusive_ptr<Storage_base> Int64Storage::to(const int &device){
        // Here, we follow pytorch scheme. if the device is the same as this->device, then return this (python self)
        // otherwise, return a clone on different device.
        if(this->device == device){
            return this;
        }else{
            if(this->device==Device.cpu){
                //here, cpu->gpu with gid=device
                #ifdef UNI_GPU
                    cytnx_error_msg(device>=Device.Ngpus,"%s","[ERROR] invalid device.");
                    cudaSetDevice(device);          
                    void *dtmp = utils_internal::cuMalloc_gpu(sizeof(cytnx_int64)*this->len);
                    checkCudaErrors(cudaMemcpy(dtmp,this->Mem,sizeof(cytnx_int64)*this->len,cudaMemcpyHostToDevice));
                    boost::intrusive_ptr<Storage_base> out(new Int64Storage());
                    out->_Init_byptr(dtmp,this->len,device);
                    return out;
                #else
                    cytnx_error_msg(1,"%s","[ERROR] try to move from cpu(Host) to gpu without CUDA support."); 
                    return nullptr;
                #endif
            }else{
                #ifdef UNI_GPU
                    if(device==Device.cpu){
                        //here, gpu->cpu
                        cudaSetDevice(this->device);
                        void *htmp = malloc(sizeof(cytnx_int64)*this->len);
                        checkCudaErrors(cudaMemcpy(htmp,this->Mem,sizeof(cytnx_int64)*this->len,cudaMemcpyDeviceToHost));
                        boost::intrusive_ptr<Storage_base> out(new Int64Storage());
                        out->_Init_byptr(htmp,this->len,device);
                        return out;
                    }else{
                        // here, gpu->gpu 
                        cytnx_error_msg(device>=Device.Ngpus,"%s","[ERROR] invalid device.");
                        cudaSetDevice(device);
                        void *dtmp = utils_internal::cuMalloc_gpu(sizeof(cytnx_int64)*this->len);
                        checkCudaErrors(cudaMemcpyPeer(dtmp,device,this->Mem,this->device,sizeof(cytnx_int64)*this->len));
                        boost::intrusive_ptr<Storage_base> out(new Int64Storage());
                        out->_Init_byptr(dtmp,this->len,device);
                        return out;
                    }
                #else
                    cytnx_error_msg(1,"%s","[ERROR][Internal] Storage.to_. the Storage is as GPU but without CUDA support.");
                    return nullptr;
                #endif
            }

        }
    }

    void Int64Storage::PrintElem_byShape(std::ostream &os, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper){
        char* buffer = (char*)malloc(sizeof(char)*256);
        // checking:
        cytnx_uint64 Ne = 1;
        for(cytnx_uint64 i=0;i<shape.size();i++){
            Ne *= shape[i];
        }
        if(Ne != this->len){
            cytnx_error_msg(1,"%s","PrintElem_byShape, the number of shape not match with the No. of elements.");
        }

        if(len==0){
            os << "[ " ;
            os << "\nThe Storage has not been allocated or linked.\n";
            os << "]\n";
        }else{

            os << std::endl << "Total elem: " << this->len << "\n";

            os << "type  : " << Type.getname(this->dtype) << std::endl;


            int atDevice = this->device;
            os << Device.getname(this->device) << std::endl;

            sprintf(buffer,"%s","Shape :"); os << string(buffer);
            sprintf(buffer," (%llu",shape[0]); os << string(buffer);
            for(cytnx_int32 i=1;i<shape.size();i++){
                sprintf(buffer,",%llu",shape[i]); os << string(buffer);
            }
            os << ")"<< std::endl;

            // temporary move to cpu for printing.
            if(this->device!=Device.cpu){
                this->to_(Device.cpu);
            }

            std::vector<cytnx_uint64> stk(shape.size(),0),stk2;


            cytnx_uint64 s;
            cytnx_int64* elem_ptr_ = static_cast<cytnx_int64*>(this->Mem);

            if(mapper.size()==0){

                cytnx_uint64 cnt=0;
                while(1){
                    for(cytnx_int32 i=0;i<shape.size();i++){
                        if(i<shape.size()-stk.size()){
                            sprintf(buffer,"%s"," "); os << string(buffer);
                        }else{
                            stk2.push_back(0);
                            sprintf(buffer,"%s","["); os << string(buffer);
                            stk.pop_back();
                        }
                    }
                    for(cytnx_uint64 i=0;i<shape.back();i++){
                        stk2.back() = i;
                        sprintf(buffer,"%+19lld ",elem_ptr_[cnt]);os << string(buffer);
                        cnt++;
                    }


                    s=0;
                    while(1){
                        if(stk2.empty()){
                            break;
                        }
                        if(stk2.back()==*(&shape.back()-s)-1){
                            stk.push_back(*(&shape.back()-s));
                            s++;
                            stk2.pop_back();
                            sprintf(buffer,"%s","]"); os << string(buffer);
                        }else{
                            stk2.back()+=1;
                            break;
                        }
                    }
                    os << "\n";

                    if(stk2.empty())break;
                }
                os << std::endl;


            }else{
                ///This is for non-contiguous Tensor printing;
                //cytnx_error_msg(1,"%s","print for a non-contiguous Storage is under developing");
                //cytnx_uint64 cnt=0;
                std::vector<cytnx_uint64> c_offj(shape.size());
                std::vector<cytnx_uint64> c_shape(shape.size());

                cytnx_uint64 accu=1;
                cytnx_uint64 RealMemPos;
                for(cytnx_uint32 i=0;i<shape.size();i++){
                    c_shape[i] = shape[mapper[i]];
                }
                for(cytnx_int64 i=c_shape.size()-1;i>=0;i--){
                    c_offj[i] = accu;
                    accu*=c_shape[i];
                }

                while(1){
                    for(cytnx_int32 i=0;i<shape.size();i++){
                        if(i<shape.size()-stk.size()){
                            sprintf(buffer,"%s"," "); os << string(buffer);
                        }else{
                            stk2.push_back(0);
                            sprintf(buffer,"%s","[");os << string(buffer);
                            stk.pop_back();
                        }
                    }
                    for(cytnx_uint64 i=0;i<shape.back();i++){
                        stk2.back() = i;

                        ///Calculate the Memory reflection:
                        RealMemPos = 0;
                        for(cytnx_uint64 n=0;n<shape.size();n++){
                            RealMemPos += c_offj[n]*stk2[mapper[n]]; // mapback + backmap = normal-map
                        }
                        sprintf(buffer,"%+19lld ",elem_ptr_[RealMemPos]);os << string(buffer);
                        //cnt++;
                    }

                    s=0;
                    while(1){
                        if(stk2.empty()){
                            break;
                        }
                        if(stk2.back()==*(&shape.back()-s)-1){
                            stk.push_back(*(&shape.back()-s));
                            s++;
                            stk2.pop_back();
                            sprintf(buffer,"%s","]");os << string(buffer);
                        }else{
                            stk2.back()+=1;
                            break;
                        }
                    }
                    os << "\n";

                    if(stk2.empty())break;
                }
                os << std::endl;


            }//check if need mapping

            if(atDevice!=Device.cpu){
                this->to_(atDevice);
            }

        }//len==0
        free(buffer);
    }
    


    void Int64Storage::print_elems(){
        char* buffer = (char*)malloc(sizeof(char)*256);
        cytnx_int64* elem_ptr_ = static_cast<cytnx_int64*>(this->Mem);
        cout << "[ ";
        for(unsigned long long cnt=0;cnt<this->len;cnt++){
            sprintf(buffer,"%+19lld ",elem_ptr_[cnt]); cout << string(buffer);
        }
        cout << " ]" << endl;
        free(buffer);
    }

    void Int64Storage::fill(const cytnx_complex128 &val){
        cytnx_error_msg(true,"[ERROR]%s"," cannot fill complex value into real container");
    }
    void Int64Storage::fill(const cytnx_complex64  &val){
        cytnx_error_msg(true,"[ERROR]%s"," cannot fill complex value into real container");
    }
    void Int64Storage::fill(const cytnx_double     &val){
        cytnx_int64 tmp = val;
        if(this->device == Device.cpu){
            utils_internal::Fill_cpu_i64(this->Mem, (void*)(&tmp), this->len);
        }else{
            #ifdef UNI_GPU
                checkCudaErrors(cudaSetDevice(this->device));
                utils_internal::cuFill_gpu_i64(this->Mem, (void*)(&tmp), this->len);
            #else
                cytnx_error_msg(true,"[ERROR][fill] fatal internal, %s","storage is on gpu without CUDA support\n");
            #endif
        }
    }
    void Int64Storage::fill(const cytnx_float      &val){
        cytnx_int64 tmp = val;
        if(this->device == Device.cpu){
            utils_internal::Fill_cpu_i64(this->Mem, (void*)(&val), this->len);
        }else{
            #ifdef UNI_GPU
                checkCudaErrors(cudaSetDevice(this->device));
                utils_internal::cuFill_gpu_i64(this->Mem, (void*)(&val), this->len);
            #else
                cytnx_error_msg(true,"[ERROR][fill] fatal internal, %s","storage is on gpu without CUDA support\n");
            #endif
        }
    }
    void Int64Storage::fill(const cytnx_int64      &val){
        if(this->device == Device.cpu){
            utils_internal::Fill_cpu_i64(this->Mem, (void*)(&val), this->len);
        }else{
            #ifdef UNI_GPU
                checkCudaErrors(cudaSetDevice(this->device));
                utils_internal::cuFill_gpu_i64(this->Mem, (void*)(&val), this->len);
            #else
                cytnx_error_msg(true,"[ERROR][fill] fatal internal, %s","storage is on gpu without CUDA support\n");
            #endif
        }
    }
    void Int64Storage::fill(const cytnx_uint64     &val){
        cytnx_int64 tmp = val;
        if(this->device == Device.cpu){
            utils_internal::Fill_cpu_i64(this->Mem, (void*)(&tmp), this->len);
        }else{
            #ifdef UNI_GPU
                checkCudaErrors(cudaSetDevice(this->device));
                utils_internal::cuFill_gpu_i64(this->Mem, (void*)(&tmp), this->len);
            #else
                cytnx_error_msg(true,"[ERROR][fill] fatal internal, %s","storage is on gpu without CUDA support\n");
            #endif
        }
    }
    void Int64Storage::fill(const cytnx_int32      &val){
        cytnx_int64 tmp = val;
        if(this->device == Device.cpu){
            utils_internal::Fill_cpu_i64(this->Mem, (void*)(&tmp), this->len);
        }else{
            #ifdef UNI_GPU
                checkCudaErrors(cudaSetDevice(this->device));
                utils_internal::cuFill_gpu_i64(this->Mem, (void*)(&tmp), this->len);
            #else
                cytnx_error_msg(true,"[ERROR][fill] fatal internal, %s","storage is on gpu without CUDA support\n");
            #endif
        }
    }
    void Int64Storage::fill(const cytnx_uint32     &val){
        cytnx_int64 tmp = val;
        if(this->device == Device.cpu){
            utils_internal::Fill_cpu_i64(this->Mem, (void*)(&tmp), this->len);
        }else{
            #ifdef UNI_GPU
                checkCudaErrors(cudaSetDevice(this->device));
                utils_internal::cuFill_gpu_i64(this->Mem, (void*)(&tmp), this->len);
            #else
                cytnx_error_msg(true,"[ERROR][fill] fatal internal, %s","storage is on gpu without CUDA support\n");
            #endif
        }
    }
    void Int64Storage::fill(const cytnx_uint16     &val){
        cytnx_int64 tmp = val;
        if(this->device == Device.cpu){
            utils_internal::Fill_cpu_i64(this->Mem, (void*)(&tmp), this->len);
        }else{
            #ifdef UNI_GPU
                checkCudaErrors(cudaSetDevice(this->device));
                utils_internal::cuFill_gpu_i64(this->Mem, (void*)(&tmp), this->len);
            #else
                cytnx_error_msg(true,"[ERROR][fill] fatal internal, %s","storage is on gpu without CUDA support\n");
            #endif
        }
    }
    void Int64Storage::fill(const cytnx_int16     &val){
        cytnx_int64 tmp = val;
        if(this->device == Device.cpu){
            utils_internal::Fill_cpu_i64(this->Mem, (void*)(&tmp), this->len);
        }else{
            #ifdef UNI_GPU
                checkCudaErrors(cudaSetDevice(this->device));
                utils_internal::cuFill_gpu_i64(this->Mem, (void*)(&tmp), this->len);
            #else
                cytnx_error_msg(true,"[ERROR][fill] fatal internal, %s","storage is on gpu without CUDA support\n");
            #endif
        }
    }
    void Int64Storage::fill(const cytnx_bool     &val){
        cytnx_int64 tmp = val;
        if(this->device == Device.cpu){
            utils_internal::Fill_cpu_i64(this->Mem, (void*)(&tmp), this->len);
        }else{
            #ifdef UNI_GPU
                checkCudaErrors(cudaSetDevice(this->device));
                utils_internal::cuFill_gpu_i64(this->Mem, (void*)(&tmp), this->len);
            #else
                cytnx_error_msg(true,"[ERROR][fill] fatal internal, %s","storage is on gpu without CUDA support\n");
            #endif
        }
    }


    void Int64Storage::set_zeros(){
        if(this->device == Device.cpu){
            utils_internal::SetZeros(this->Mem,sizeof(cytnx_int64)*this->len);
        }else{
            #ifdef UNI_GPU
                checkCudaErrors(cudaSetDevice(this->device));
                utils_internal::cuSetZeros(this->Mem,sizeof(cytnx_int64)*this->len);
            #else
                cytnx_error_msg(1,"[ERROR][set_zeros] fatal, the storage is on gpu without CUDA support.%s","\n");
            #endif
        }
    }


}//cytnx
