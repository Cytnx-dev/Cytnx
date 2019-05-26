#include "Storage.hpp"
#include "utils/utils_internal.hpp"

using namespace std;

namespace cytnx{
    void Int32Storage::Init(const unsigned long long &len_in,const int &device){
        //cout << "Int32.init" << endl;
        this->len = len_in;

        //check:
        cytnx_error_msg(len_in < 1, "%s", "[ERROR] cannot init a Storage with zero element");
        this->dtype = cytnxtype.Int32;
        if(device==cytnxdevice.cpu){
            this->Mem = utils_internal::Calloc_cpu(this->len,sizeof(cytnx_int32));
        }else{
            #ifdef UNI_GPU
                cytnx_error_msg(device>=cytnxdevice.Ngpus,"%s","[ERROR] invalid device.");
                cudaSetDevice(device);
                this->Mem = utils_internal::cuMalloc_gpu(this->len*sizeof(cytnx_int32));
            #else
                cytnx_error_msg(1,"%s","[ERROR] cannot init a Storage on gpu without CUDA support.");
            #endif
        }
        this->device = device;
    }

    void Int32Storage::_Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device){
        this->Mem = rawptr;
        this->len = len_in;
    # ifdef UNI_DEBUG
        cytnx_error_msg(len_in < 1, "%s", "[ERROR] _Init_by_ptr cannot have len_in < 1.");
    # endif
        this->dtype = cytnxtype.Int32;
        this->device = device;
    }

    boost::intrusive_ptr<Storage_base> Int32Storage::_create_new_sametype(){
        boost::intrusive_ptr<Storage_base> out(new Int32Storage());
        return out;
    }

    boost::intrusive_ptr<Storage_base> Int32Storage::copy(){
        boost::intrusive_ptr<Storage_base> out(new Int32Storage());
        out->Init(this->len,this->device);
        if(this->device==cytnxdevice.cpu){
            memcpy(out->Mem,this->Mem,sizeof(cytnx_int32)*this->len);
        }else{
            #ifdef UNI_GPU
                checkCudaErrors(cudaMemcpy(out->Mem,this->Mem,sizeof(cytnx_int32)*this->len,cudaMemcpyDeviceToDevice));
            #else
                cytnx_error_msg(1,"%s","[ERROR] cannot copy a Storage on gpu without CUDA support.");
            #endif
        }
        return out;
    }


    void Int32Storage::Move_memory_(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper){
        boost::intrusive_ptr<Storage_base> tmp(this);
        if(this->device==cytnxdevice.cpu){
            utils_internal::Movemem_cpu_i32(tmp,old_shape,mapper,invmapper,1);
        }else{
            #ifdef UNI_GPU
                utils_internal::cuMovemem_gpu_i32(tmp,old_shape,mapper,invmapper,1); 
            #else
                cytnx_error_msg(1,"%s","[ERROR][Internal] try to call GPU section without CUDA support");
            #endif
        }

    }

    boost::intrusive_ptr<Storage_base> Int32Storage::Move_memory(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper){
        boost::intrusive_ptr<Storage_base> tmp(this);
        if(this->device==cytnxdevice.cpu){
            return utils_internal::Movemem_cpu_i32(tmp,old_shape,mapper,invmapper,0);
        }else{
            #ifdef UNI_GPU
                return utils_internal::cuMovemem_gpu_i32(tmp,old_shape,mapper,invmapper,0); 
            #else
                cytnx_error_msg(1,"%s","[ERROR][Internal] try to call GPU section without CUDA support");
            #endif
        }


    }
    void Int32Storage::to_(const int &device){
        if(this->device != device){
            if(this->device==cytnxdevice.cpu){
                //here, cpu->gpu with gid=device
                #ifdef UNI_GPU
                    cytnx_error_msg(device>=cytnxdevice.Ngpus,"%s","[ERROR] invalid device.");
                    cudaSetDevice(device);          
                    void *dtmp = utils_internal::cuMalloc_gpu(sizeof(cytnx_int32)*this->len);
                    checkCudaErrors(cudaMemcpy(dtmp,this->Mem,sizeof(cytnx_int32)*this->len,cudaMemcpyHostToDevice));
                    free(this->Mem);
                    this->Mem = dtmp;
                    this->device=device;
                #else
                    cytnx_error_msg(1,"%s","[ERROR] try to move from cpu(Host) to gpu without CUDA support."); 
                #endif
            }else{
                #ifdef UNI_GPU
                    if(device==cytnxdevice.cpu){
                        //here, gpu->cpu
                        cudaSetDevice(this->device);
                        void *htmp = malloc(sizeof(cytnx_int32)*this->len);
                        checkCudaErrors(cudaMemcpy(htmp,this->Mem,sizeof(cytnx_int32)*this->len,cudaMemcpyDeviceToHost));
                        cudaFree(this->Mem);
                        this->Mem = htmp;
                        this->device = device;
                    }else{
                        // here, gpu->gpu 
                        cytnx_error_msg(device>=cytnxdevice.Ngpus,"%s","[ERROR] invalid device.");
                        cudaSetDevice(device);
                        void *dtmp = utils_internal::cuMalloc_gpu(sizeof(cytnx_int32)*this->len);
                        checkCudaErrors(cudaMemcpyPeer(dtmp,device,this->Mem,this->device,sizeof(cytnx_int32)*this->len));
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
    boost::intrusive_ptr<Storage_base> Int32Storage::to(const int &device){
        // Here, we follow pytorch scheme. if the device is the same as this->device, then return this (python self)
        // otherwise, return a copy on different device.
        if(this->device == device){
            return this;
        }else{
            if(this->device==cytnxdevice.cpu){
                //here, cpu->gpu with gid=device
                #ifdef UNI_GPU
                    cytnx_error_msg(device>=cytnxdevice.Ngpus,"%s","[ERROR] invalid device.");
                    cudaSetDevice(device);          
                    void *dtmp = utils_internal::cuMalloc_gpu(sizeof(cytnx_int32)*this->len);
                    checkCudaErrors(cudaMemcpy(dtmp,this->Mem,sizeof(cytnx_int32)*this->len,cudaMemcpyHostToDevice));
                    boost::intrusive_ptr<Storage_base> out(new Int32Storage());
                    out->_Init_byptr(dtmp,this->len,device);
                    return out;
                #else
                    cytnx_error_msg(1,"%s","[ERROR] try to move from cpu(Host) to gpu without CUDA support."); 
                #endif
            }else{
                #ifdef UNI_GPU
                    if(device==cytnxdevice.cpu){
                        //here, gpu->cpu
                        cudaSetDevice(this->device);
                        void *htmp = malloc(sizeof(cytnx_int32)*this->len);
                        checkCudaErrors(cudaMemcpy(htmp,this->Mem,sizeof(cytnx_int32)*this->len,cudaMemcpyDeviceToHost));
                        boost::intrusive_ptr<Storage_base> out(new Int32Storage());
                        out->_Init_byptr(htmp,this->len,device);
                        return out;
                    }else{
                        // here, gpu->gpu 
                        cytnx_error_msg(device>=cytnxdevice.Ngpus,"%s","[ERROR] invalid device.");
                        cudaSetDevice(device);
                        void *dtmp = utils_internal::cuMalloc_gpu(sizeof(cytnx_int32)*this->len);
                        checkCudaErrors(cudaMemcpyPeer(dtmp,device,this->Mem,this->device,sizeof(cytnx_int32)*this->len));
                        boost::intrusive_ptr<Storage_base> out(new Int32Storage());
                        out->_Init_byptr(dtmp,this->len,device);
                        return out;
                    }
                #else
                    cytnx_error_msg(1,"%s","[ERROR][Internal] Storage.to_. the Storage is as GPU but without CUDA support.");
                #endif
            }

        }
    }

    void Int32Storage::PrintElem_byShape(std::ostream &os, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper){
    
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

            os << "type  : " << cytnxtype.getname(this->dtype) << std::endl;


            int atDevice = this->device;
            os << cytnxdevice.getname(this->device) << std::endl;

            printf("%s","Shape :");
            printf(" (%d",shape[0]);
            for(cytnx_int32 i=1;i<shape.size();i++){
                printf(",%d",shape[i]);
            }
            os << ")"<< std::endl;

            // temporary move to cpu for printing.
            if(this->device!=cytnxdevice.cpu){
                this->to_(cytnxdevice.cpu);
            }

            std::vector<cytnx_uint64> stk(shape.size(),0),stk2;


            cytnx_uint64 s;
            cytnx_int32* elem_ptr_ = static_cast<cytnx_int32*>(this->Mem);

            if(mapper.size()==0){

                cytnx_uint64 cnt=0;
                while(1){
                    for(cytnx_int32 i=0;i<shape.size();i++){
                        if(i<shape.size()-stk.size()){
                            printf("%s"," ");
                        }else{
                            stk2.push_back(0);
                            printf("%s","[");
                            stk.pop_back();
                        }
                    }
                    for(cytnx_uint64 i=0;i<shape.back();i++){
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
                            printf("%s"," ");
                        }else{
                            stk2.push_back(0);
                            printf("%s","[");
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

            if(atDevice!=cytnxdevice.cpu){
                this->to_(atDevice);
            }

        }//len==0
    }

    void Int32Storage::print_elems(){
        cytnx_int32* elem_ptr_ = static_cast<cytnx_int32*>(this->Mem);
        cout << "[ ";
        for(unsigned long long cnt=0;cnt<this->len;cnt++){
            printf("%+10d ",elem_ptr_[cnt]);
        }
        cout << " ]" << endl;
    }

}//cytnx
