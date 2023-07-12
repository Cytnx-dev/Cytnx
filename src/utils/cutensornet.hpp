#ifndef __cutensornet_H_
#define __cutensornet_H_

#include "Type.hpp"
#include "cytnx_error.hpp"

namespace cytnx {

#ifdef UNI_GPU

  void cuTensornet_(const int32_t numInputs, void* R_d, int32_t nmodeR, int64_t* extentR,
                    int32_t* modesR, void* rawDataIn_d[], int32_t* modesIn[],
                    int32_t const numModesIn[], int64_t* extentsIn[], int64_t* stridesIn[],
                    bool verbose);
  void callcuTensornet(UniTensor& res, std::vector<UniTensor>& uts, bool& verbose);

#endif

}  // namespace cytnx
#endif
