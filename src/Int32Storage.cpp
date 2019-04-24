#include "Storage.hpp"
#include "utils/utils_internal.hpp"

using namespace std;

namespace tor10{
    void Int32Storage::Init(const unsigned long long &len_in,const int &device){
        //cout << "Int32.init" << endl;
        this->len = len_in;

        //check:
        tor10_error_msg(len_in < 1, "%s", "[ERROR] cannot init a Storage with zero element");
        this->dtype_id = tor10type.Int32;
        if(device==tor10device.cpu){
            this->Mem = utils_internal::Calloc_cpu(this->len,sizeof(tor10_int32));
        }else{
            #ifdef UNI_GPU
                tor10_error_msg(device>=tor10device.Ngpus,"%s","[ERROR] invalid device.");
                cudaSetDevice(device);
                this->Mem = utils_internal::cuMalloc_gpu(this->len*sizeof(tor10_int32));
            #else
                tor10_error_msg(1,"%s","[ERROR] cannot init a Storage on gpu without CUDA support.");
            #endif
        }
        this->device = device;
    }

    void Int32Storage::_Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device){
        this->Mem = rawptr;
        this->len = len_in;
    # ifdef UNI_DEBUG
        tor10_error_msg(len_in < 1, "%s", "[ERROR] _Init_by_ptr cannot have len_in < 1.");
    # endif
        this->dtype_id = tor10type.Int32;
        this->device = device;
    }

    boost::intrusive_ptr<Storage_base> Int32Storage::_create_new_sametype(){
        boost::intrusive_ptr<Storage_base> out(new Int32Storage());
        return out;
    }

    boost::intrusive_ptr<Storage_base> Int32Storage::copy(){
        boost::intrusive_ptr<Storage_base> out(new Int32Storage());
        out->Init(this->len,this->device);
        if(this->device==tor10device.cpu){
            memcpy(out->Mem,this->Mem,sizeof(tor10_int32)*this->len);
        }else{
            #ifdef UNI_GPU
                checkCudaErrors(cudaMemcpy(out->Mem,this->Mem,sizeof(tor10_int32)*this->len,cudaMemcpyDeviceToDevice));
            #else
                tor10_error_msg(1,"%s","[ERROR] cannot copy a Storage on gpu without CUDA support.");
            #endif
        }
        return out;
    }


    void Int32Storage::Move_memory_(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper){
        boost::intrusive_ptr<Storage_base> tmp(this);
        if(this->device==tor10device.cpu){
            utils_internal::Movemem_cpu_i32(tmp,old_shape,mapper,invmapper,1);
        }else{
            #ifdef UNI_GPU
                utils_internal::cuMovemem_gpu_i32(tmp,old_shape,mapper,invmapper,1); 
            #else
                tor10_error_msg(1,"%s","[ERROR][Internal] try to call GPU section without CUDA support");
            #endif
        }

    }

    boost::intrusive_ptr<Storage_base> Int32Storage::Move_memory(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper){
        boost::intrusive_ptr<Storage_base> tmp(this);
        if(this->device==tor10device.cpu){
            return utils_internal::Movemem_cpu_i32(tmp,old_shape,mapper,invmapper,0);
        }else{
            #ifdef UNI_GPU
                return utils_internal::cuMovemem_gpu_i32(tmp,old_shape,mapper,invmapper,0); 
            #else
                tor10_error_msg(1,"%s","[ERROR][Internal] try to call GPU section without CUDA support");
            #endif
        }


    }
    void Int32Storage::to_(const int &device){
        if(this->device != device){
            if(this->device==tor10device.cpu){
                //here, cpu->gpu with gid=device
                #ifdef UNI_GPU
                    tor10_error_msg(device>=tor10device.Ngpus,"%s","[ERROR] invalid device.");
                    cudaSetDevice(device);          
                    void *dtmp = utils_internal::cuMalloc_gpu(sizeof(tor10_int32)*this->len);
                    checkCudaErrors(cudaMemcpy(dtmp,this->Mem,sizeof(tor10_int32)*this->len,cudaMemcpyHostToDevice));
                    free(this->Mem);
                    this->Mem = dtmp;
                    this->device=device;
                #else
                    tor10_error_msg(1,"%s","[ERROR] try to move from cpu(Host) to gpu without CUDA support."); 
                #endif
            }else{
                #ifdef UNI_GPU
                    if(device==tor10device.cpu){
                        //here, gpu->cpu
                        cudaSetDevice(this->device);
                        void *htmp = malloc(sizeof(tor10_int32)*this->len);
                        checkCudaErrors(cudaMemcpy(htmp,this->Mem,sizeof(tor10_int32)*this->len,cudaMemcpyDeviceToHost));
                        cudaFree(this->Mem);
                        this->Mem = htmp;
                        this->device = device;
                    }else{
                        // here, gpu->gpu 
                        tor10_error_msg(device>=tor10device.Ngpus,"%s","[ERROR] invalid device.");
                        cudaSetDevice(device);
                        void *dtmp = utils_internal::cuMalloc_gpu(sizeof(tor10_int32)*this->len);
                        checkCudaErrors(cudaMemcpyPeer(dtmp,device,this->Mem,this->device,sizeof(tor10_int32)*this->len));
                        cudaFree(this->Mem);
                        this->Mem = dtmp;
                        this->device = device;
                    }
                #else
                    tor10_error_msg(1,"%s","[ERROR][Internal] Storage.to_. the Storage is as GPU but without CUDA support.");
                #endif
            }
        }
    }
    boost::intrusive_ptr<Storage_base> Int32Storage::to(const int &device){
        // Here, we follow pytorch scheme. if the device is the same as this->device, then return this (python self)
        // otherwise, return a copy on different device.
        if(this->device == device){
            return this;
        }else{
            if(this->device==tor10device.cpu){
                //here, cpu->gpu with gid=device
                #ifdef UNI_GPU
                    tor10_error_msg(device>=tor10device.Ngpus,"%s","[ERROR] invalid device.");
                    cudaSetDevice(device);          
                    void *dtmp = utils_internal::cuMalloc_gpu(sizeof(tor10_int32)*this->len);
                    checkCudaErrors(cudaMemcpy(dtmp,this->Mem,sizeof(tor10_int32)*this->len,cudaMemcpyHostToDevice));
                    boost::intrusive_ptr<Storage_base> out(new Int32Storage());
                    out->_Init_byptr(dtmp,this->len,device);
                    return out;
                #else
                    tor10_error_msg(1,"%s","[ERROR] try to move from cpu(Host) to gpu without CUDA support."); 
                #endif
            }else{
                #ifdef UNI_GPU
                    if(device==tor10device.cpu){
                        //here, gpu->cpu
                        cudaSetDevice(this->device);
                        void *htmp = malloc(sizeof(tor10_int32)*this->len);
                        checkCudaErrors(cudaMemcpy(htmp,this->Mem,sizeof(tor10_int32)*this->len,cudaMemcpyDeviceToHost));
                        boost::intrusive_ptr<Storage_base> out(new Int32Storage());
                        out->_Init_byptr(htmp,this->len,device);
                        return out;
                    }else{
                        // here, gpu->gpu 
                        tor10_error_msg(device>=tor10device.Ngpus,"%s","[ERROR] invalid device.");
                        cudaSetDevice(device);
                        void *dtmp = utils_internal::cuMalloc_gpu(sizeof(tor10_int32)*this->len);
                        checkCudaErrors(cudaMemcpyPeer(dtmp,device,this->Mem,this->device,sizeof(tor10_int32)*this->len));
                        boost::intrusive_ptr<Storage_base> out(new Int32Storage());
                        out->_Init_byptr(dtmp,this->len,device);
                        return out;
                    }
                #else
                    tor10_error_msg(1,"%s","[ERROR][Internal] Storage.to_. the Storage is as GPU but without CUDA support.");
                #endif
            }

        }
    }

    void Int32Storage::PrintElem_byShape(std::ostream &os, const std::vector<tor10_uint64> &shape, const std::vector<tor10_uint64> &mapper){
    
        // checking:
        tor10_uint64 Ne = 1;
        for(tor10_uint64 i=0;i<shape.size();i++){
            Ne *= shape[i];
        }
        if(Ne != this->len){
            tor10_error_msg(1,"%s","PrintElem_byShape, the number of shape not match with the No. of elements.");
        }

        if(len==0){
            os << "[ " ;
            os << "\nThe Storage has not been allocated or linked.\n";
            os << "]\n";
        }else{

            os << std::endl << "Total elem: " << this->len << "\n";

            os << "type  : " << tor10type.getname(this->dtype_id) << std::endl;


            int atDevice = this->device;
            os << tor10device.getname(this->device) << std::endl;

            printf("%s","Shape :");
            printf(" (%d",shape[0]);
            for(tor10_int32 i=1;i<shape.size();i++){
                printf(",%d",shape[i]);
            }
            os << ")"<< std::endl;

            // temporary move to cpu for printing.
            if(this->device!=tor10device.cpu){
                this->to_(tor10device.cpu);
            }

            std::vector<tor10_uint64> stk(shape.size(),0),stk2;


            tor10_uint64 s;
            tor10_int32* elem_ptr_ = static_cast<tor10_int32*>(this->Mem);

            if(mapper.size()==0){

                tor10_uint64 cnt=0;
                while(1){
                    for(tor10_int32 i=0;i<shape.size();i++){
                        if(i<shape.size()-stk.size()){
                            printf("%s"," ");
                        }else{
                            stk2.push_back(0);
                            printf("%s","[");
                            stk.pop_back();
                        }
                    }
                    for(tor10_uint64 i=0;i<shape.back();i++){
                        stk2.back() = i;
                        printf("%+10d ",elem_ptr_[cnt]);
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
                            printf("%s","]");
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
                //tor10_error_msg(1,"%s","print for a non-contiguous Storage is under developing");
                //tor10_uint64 cnt=0;
                std::vector<tor10_uint64> c_offj(shape.size());
                std::vector<tor10_uint64> c_shape(shape.size());

                tor10_uint64 accu=1;
                tor10_uint64 RealMemPos;
                for(tor10_uint32 i=0;i<shape.size();i++){
                    c_shape[i] = shape[mapper[i]];
                }
                for(tor10_int64 i=c_shape.size()-1;i>=0;i--){
                    c_offj[i] = accu;
                    accu*=c_shape[i];
                }

                while(1){
                    for(tor10_int32 i=0;i<shape.size();i++){
                        if(i<shape.size()-stk.size()){
                            printf("%s"," ");
                        }else{
                            stk2.push_back(0);
                            printf("%s","[");
                            stk.pop_back();
                        }
                    }
                    for(tor10_uint64 i=0;i<shape.back();i++){
                        stk2.back() = i;

                        ///Calculate the Memory reflection:
                        RealMemPos = 0;
                        for(tor10_uint64 n=0;n<shape.size();n++){
                            RealMemPos += c_offj[n]*stk2[mapper[n]]; // mapback + backmap = normal-map
                        }
                        printf("%+10d ",elem_ptr_[RealMemPos]);
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
                            printf("%s","]");
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

            if(atDevice!=tor10device.cpu){
                this->to_(atDevice);
            }

        }//len==0
    }
}//tor10
