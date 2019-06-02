#include "UniTensor.hpp"

namespace cytnx{


        void UniTensor_base::Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels, const cytnx_int64 &Rowrank,const unsigned int &dtype,const int &device,const bool &is_diag){
            cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s","\n");
        }

        std::vector<cytnx_uint64> UniTensor_base::shape() const{
            cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s","\n");
        }
        bool      UniTensor_base::is_blockform() const{
            cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s","\n");
        }
        bool     UniTensor_base::is_contiguous() const{
            cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s","\n");
        }
        void UniTensor_base::to_(const int &device){
            cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s","\n");
        }
        boost::intrusive_ptr<UniTensor_base> UniTensor_base::to(const int &device) const{
            cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s","\n");
        }
        boost::intrusive_ptr<UniTensor_base> UniTensor_base::clone() const{
            cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s","\n");
        }
        unsigned int  UniTensor_base::dtype() const{
            cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s","\n");
        }
       int          UniTensor_base::device() const{
            cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s","\n");
        }
        std::string      UniTensor_base::dtype_str() const{
            cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s","\n");
        }
        std::string     UniTensor_base::device_str() const{
            cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s","\n");
        }


}

