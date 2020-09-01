#ifndef _H_TORCYX_
#define _H_TORCYX_

//#include "cytnx.hpp"
#include "Symmetry.hpp";
#include "Bond.hpp";
#include "Accessor.hpp";
#include "intrusive_ptr_base.hpp";
#include "Type.hpp"
#include "Device.hpp"
namespace torcyx{
    using cytnx::cytnx_double;
    using cytnx::cytnx_float;
    using cytnx::cytnx_uint64;
    using cytnx::cytnx_uint32;
    using cytnx::cytnx_uint16;
    using cytnx::cytnx_int64;
    using cytnx::cytnx_int32;
    using cytnx::cytnx_int16;
    using cytnx::cytnx_size_t;
    using cytnx::cytnx_complex64;
    using cytnx::cytnx_complex128;
    using cytnx::cytnx_bool;


    using cytnx::intrusive_ptr_base;
    using cytnx::Symmetry;
    using cytnx::Bond;
    using cytnx::bondType;
    using cytnx::Accessor;
};

#include "ml/TypeConvert.hpp"
#include "ml/CyTensor.hpp"
#include "ml/xlinalg.hpp"

#endif
