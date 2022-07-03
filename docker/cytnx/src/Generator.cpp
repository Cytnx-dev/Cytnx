#include "Generator.hpp"
#include "Storage.hpp"
#include "utils/utils.hpp"
#include "utils/utils_internal_interface.hpp"
#include <cfloat>
#include <iostream>
namespace cytnx {

  Tensor zeros(const cytnx_uint64 &Nelem, const unsigned int &dtype, const int &device) {
    Tensor out({Nelem}, dtype, device);  // the default
    out._impl->storage().set_zeros();
    return out;
  }
  Tensor zeros(const std::vector<cytnx_uint64> &Nelem, const unsigned int &dtype,
               const int &device) {
    // std::cout << "OK" << std::endl;
    // std::cout << Nelem << std::endl;
    Tensor out(Nelem, dtype, device);
    out._impl->storage().set_zeros();
    return out;
  }
  //-----------------
  Tensor ones(const cytnx_uint64 &Nelem, const unsigned int &dtype, const int &device) {
    Tensor out({Nelem}, dtype, device);  // the default
    out._impl->storage().fill(1);
    return out;
  }
  Tensor ones(const std::vector<cytnx_uint64> &Nelem, const unsigned int &dtype,
              const int &device) {
    Tensor out(Nelem, dtype, device);
    out._impl->storage().fill(1);
    return out;
  }
  //-----------------
  Tensor arange(const cytnx_double &start, const cytnx_double &end, const cytnx_double &step,
                const unsigned int &dtype, const int &device) {
    cytnx_uint64 Nelem;
    Tensor out;
    if (start < end) {
      Nelem = cytnx_uint64((end - start) / step);
      if (fmod((end - start), step) > 1.0e-14) Nelem += 1;
    } else {
      Nelem = cytnx_uint64((start - end) / (-step));
      if (fmod((start - end), (-step)) > 1.0e-14) Nelem += 1;
    }
    cytnx_error_msg(Nelem == 0, "[ERROR] arange(start,end,step)%s",
                    "Nelem cannot be zero! check the range!\n");
    out.Init({Nelem}, dtype, device);

    if (device == Device.cpu) {
      utils_internal::uii.SetArange_ii[dtype](out._impl->storage()._impl, start, end, step, Nelem);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(out.device()));
      utils_internal::uii.cuSetArange_ii[dtype](out._impl->storage()._impl, start, end, step,
                                                Nelem);
#else
      cytnx_error_msg(true, "[ERROR] fatal internal, %s",
                      " [arange] the container is on gpu without CUDA support!%s", "\n")
#endif
    }

    return out;
  }
  Tensor arange(const cytnx_int64 &Nelem, const unsigned int &dtype, const int &device) {
    cytnx_error_msg(Nelem <= 0, "[ERROR] arange(Nelem) , %s", "Nelem must be integer > 0");
    return arange(0, Nelem, 1, dtype, device);
  }
  //--------------

}  // namespace cytnx
