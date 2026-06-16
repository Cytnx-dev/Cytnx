#ifndef CYTNX_LINOP_H_
#define CYTNX_LINOP_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include <vector>
#include <string>
#include "intrusive_ptr_base.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else
namespace cytnx {

  class LinOp {
   private:
    // dimension of the operator's domain (length of the input vector)
    cytnx_uint64 _nx;

    // device and dtype
    int _device;
    int _dtype;

   public:
    /**
    @brief Linear Operator class for iterative solvers.
    @param nx the dimension of the operator's domain (the length of the input vector).
    @param dtype the Operator's dtype. Note that this should match the input/output Tensor's dtype.
    @param device the Operator's on device.

    ## Note:
        1. the device and dtype should be set. This should be the same as the input and output
    vectors. By default, we assume the input and output vectors are on CPU and of Double type.

    ## Details:
        The LinOp class defines a custom linear operation acting on a Tensor or UniTensor. To use,
    inherit this class and override the matvec function. See the following examples for how to use
    them.

    ## Example:
    ### python API:
    \include example/LinOp/init.py
    #### output>
    \verbinclude example/LinOp/init.py.out

    */
    LinOp(const cytnx_uint64 &nx, const int &dtype = Type.Double, const int &device = Device.cpu) {
      this->_nx = nx;
      cytnx_error_msg(device < -1 || device >= Device.Ngpus, "[ERROR] invalid device.%s", "\n");
      this->_device = device;
      cytnx_error_msg(dtype < 1 || dtype >= N_Type, "[ERROR] invalid dtype.%s", "\n");
      this->_dtype = dtype;
    };

    /**
    @brief Backwards-compatible constructor accepting the legacy \p type argument.
    @param type legacy operator type; must be "mv".
    @param nx the dimension of the operator's domain (the length of the input vector).
    @param dtype the Operator's dtype.
    @param device the Operator's on device.

    @deprecated Retained only for source-level backwards compatibility and not used anywhere in
    Cytnx itself. Use LinOp(nx, dtype, device) instead. \p type must equal "mv" (the "mv_elem" path
    has been removed); any other value is a hard error.
    */
    [[deprecated(
      "The type argument is obsolete; use LinOp(nx, dtype, device) instead (type must be "
      "\"mv\").")]] LinOp(const std::string &type, const cytnx_uint64 &nx,
                          const int &dtype = Type.Double, const int &device = Device.cpu)
        : LinOp(nx, dtype, device) {
      cytnx_error_msg(type != "mv",
                      "[ERROR][LinOp] the only supported type is \"mv\"; the \"mv_elem\" path has "
                      "been removed. Use LinOp(nx, ...) and override matvec().%s",
                      "\n");
    };

    void set_device(const int &device) {
      cytnx_error_msg(device < -1 || device >= Device.Ngpus, "[ERROR] invalid device.%s", "\n");
      this->_device = device;
    };
    void set_dtype(const int &dtype) {
      cytnx_error_msg(dtype < 1 || dtype >= N_Type, "[ERROR] invalid dtype.%s", "\n");
      this->_dtype = dtype;
    };
    int device() const { return this->_device; };
    int dtype() const { return this->_dtype; };
    cytnx_uint64 nx() const { return this->_nx; };

    /// @cond
    // this expose to interitance:
    // need user to check the output to be Tensor
    /// @endcond
    virtual Tensor matvec(const Tensor &Tin);

    /// @cond
    // this expose to interface:
    virtual UniTensor matvec(const UniTensor &Tin);
    // virtual std::vector<UniTensor> matvec(const std::vector<UniTensor> &Tin);
    /// @endcond
  };

}  // namespace cytnx

#endif  // BACKEND_TORCH

#endif  // CYTNX_LINOP_H_
