#ifndef __cytnx_H_
#define __cytnx_H_
#include <iostream>
#include <typeinfo>

#include "Accessor.hpp"
#include "Device.hpp"
#include "Type.hpp"

#ifdef BACKEND_TORCH
  #include "backend_torch/Type_convert.hpp"  // maybe we dont need this?
#else
  #include "backend/Storage.hpp"
#endif

#include "Tensor.hpp"
#include "Generator.hpp"
#include "Physics.hpp"
#include "algo.hpp"
#include "stat.hpp"
#include "linalg.hpp"
#include "cytnx.hpp"
#include "utils/utils.hpp"
#include "random.hpp"
#include "ncon.hpp"

#include "UniTensor.hpp"
#include "Symmetry.hpp"
#include "Network.hpp"
#include "Bond.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/Scalar.hpp"
#endif

#include "LinOp.hpp"
#include "utils/is.hpp"
#include "utils/print.hpp"

#include "tn_algo/MPS.hpp"
#include "tn_algo/MPO.hpp"
#include "tn_algo/DMRG.hpp"

#include "Gncon.hpp"

#endif
