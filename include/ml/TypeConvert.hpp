#ifndef _TYPECONVERT_H_
#define _TYPECONVERT_H_


#include "Type.hpp"
#include "Device.hpp"
#include <torch/torch.h>

namespace torcyx{
    using cytnx::Type;
    using cytnx::Device;

    /// @cond
        //typedef torch::TensorOptions (*Tor2Cy_io)(const unsigned int &dtype, const unsigned int &device);

        class TypeCvrt_class{
            public:

                //Cast
                //std::vector<Tor2Cy_io> _t2c;            
                TypeCvrt_class();
                torch::TensorOptions Cy2Tor(const unsigned int &dtype, const int &device);
                unsigned int Tor2Cy(const torch::ScalarType &scalar_type);

        };
        extern TypeCvrt_class type_converter;
    /// @endcond

}



#endif
