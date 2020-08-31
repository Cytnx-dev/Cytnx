#include "ml/TypeConvert.hpp"
#include "cytnx_error.hpp"
using namespace std;
namespace cytnx{
    namespace ml{

        TypeCvrt_class::TypeCvrt_class(){

            //_t2c = vector<Tor2Cy_io>(N_Type);

        }
        
        torch::TensorOptions TypeCvrt_class::Cy2Tor(const unsigned int &dtype, const int &device){
            auto options = torch::TensorOptions();
            if(device<0){
                options.device(torch::kCPU);
            }else{
                options.device(torch::kCUDA,device);
            }

            switch(dtype){
                case Type.Double:
                    return options.dtype(torch::kFloat64);                
                case Type.Float:
                    return options.dtype(torch::kFloat32);                
                case Type.Int64:
                    return options.dtype(torch::kInt64);
                case Type.Int32:
                    return options.dtype(torch::kInt32);
                case Type.Int16:
                    return options.dtype(torch::kInt16);
                case Type.Uint16:
                    cytnx_error_msg(true,"[ERROR] Torch type does not have Uint16.%s","\n");
                    return options;                
                case Type.Uint32:
                    cytnx_error_msg(true,"[ERROR] Torch type does not have Uint32.%s","\n");
                    return options;                
                case Type.Uint64:
                    cytnx_error_msg(true,"[ERROR] Torch type does not have Uint64.%s","\n");
                    return options;                
                case Type.Bool:
                    cytnx_error_msg(true,"[ERROR] Torch type does not have Bool.%s","\n");
                    return options;                
                case Type.Void:
                    cytnx_error_msg(true,"[ERROR] Torch type does not have Void.%s","\n");
                    return options;                
                case Type.ComplexDouble:
                    cytnx_error_msg(true,"[ERROR] Torch type does not support Complex64.%s","\n");
                    return options;                
                case Type.ComplexFloat:
                    cytnx_error_msg(true,"[ERROR] Torch type does not support Complex32.%s","\n");
                    return options;                

            };

        }

        TypeCvrt_class type_converter;

    }
}


