#ifndef CYTNX_LINOP_H_
#define CYTNX_LINOP_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include <vector>
#include <fstream>
#include <functional>
#include <map>
#include <utility>
#include <algorithm>
#include "intrusive_ptr_base.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else
namespace cytnx {

  class LinOp {
   private:
    // type:
    std::string _type;

    // nx
    cytnx_uint64 _nx;

    // device
    int _device;
    int _dtype;

    // pre-storage data:
    std::map<cytnx_uint64, std::pair<std::vector<cytnx_uint64>, Tensor>>
      _elems;  // map[i] -> pair[<js>,<Storage>]
    std::map<cytnx_uint64, std::pair<std::vector<cytnx_uint64>, Tensor>>::iterator _elems_it;

    Tensor _mv_elemfunc(const Tensor &);

   public:
    /// @cond
    // we need driver of void f(nx,vin,vout)
    /// @endcond

    /**
    @brief Linear Operator class for iterative solvers.
    @param type the type of operator. Use "mv" (matvec) and override matvec(). The "mv_elem"
    (matvec with pre-stored elements) type is **deprecated**; see the note below.
    @param nx the last dimension of operator, this should be the dimension of the input vector when
    "mv_elem" is used.
    @param dtype the Operator's dtype. Note that this should match the input/output Tensor's dtype.
    @param device the Operator's on device.

    ## Note:
        1. the device and dtype should be set. This should be the same as the input and output
    vectors. by default, we assume custom_f take input and output vector to be on CPU and Double
    type.

    @deprecated The "mv_elem" type, together with set_elem() and operator()(i, j), is deprecated
    and will be removed in a future release. It stores individual non-zero elements and computes
    the matrix-vector product element-by-element, which is far slower than a hand-written matvec.
    Instead, construct with type "mv" and override matvec() to implement the linear map directly
    (which can use any sparse structure you like).

    ## Details:
        The LinOp class is a class that defines a custom Linear operation acting on a Tensor or
    UniTensor. To use, inherit this class and override the matvec function. See the following
    examples for how to use them.

    ## Example:
    ### python API:
    \include example/LinOp/init.py
    #### output>
    \verbinclude example/LinOp/init.py.out

    */
    LinOp(const std::string &type, const cytnx_uint64 &nx, const int &dtype = Type.Double,
          const int &device = Device.cpu) {
      if (type == "mv") {
      } else if (type == "mv_elem") {
        cytnx_warning_msg(
          true,
          "[DEPRECATED][LinOp] the \"mv_elem\" type is deprecated and will be removed in a future "
          "release. Construct with type \"mv\" and override matvec() instead.%s",
          "\n");
      } else
        cytnx_error_msg(type != "mv",
                        "[ERROR][LinOp] currently only type=\"mv\" (matvec) can be used.%s", "\n");

      this->_type = type;
      this->_nx = nx;
      cytnx_error_msg(device < -1 || device >= Device.Ngpus, "[ERROR] invalid device.%s", "\n");
      this->_device = device;
      cytnx_error_msg(dtype < 1 || dtype >= N_Type, "[ERROR] invalid dtype.%s", "\n");
      this->_dtype = dtype;
    };
    /*
    void set_func(std::function<Tensor(const Tensor&)> custom_f, const int &dtype, const int
    &device){ if(this->_type=="mv"){ this->_mvfunc = custom_f; cytnx_error_msg(device<-1 || device
    >=Device.Ngpus,"[ERROR] invalid device.%s","\n"); this->_device = device;
            cytnx_error_msg(dtype<1 || dtype >= N_Type,"[ERROR] invalid dtype.%s","\n");
            this->_dtype = dtype;
        }else{
            cytnx_error_msg(true,"[ERROR] Cannot specify func with type=mv_elem%s. use set_elem
    instead.","\n");
        }
    };
    */
    /**
    @deprecated Part of the deprecated "mv_elem" path. Override matvec() with a type "mv" LinOp
    instead.
    */
    template <class T>
    [[deprecated(
      "The \"mv_elem\" path (set_elem) is deprecated; construct a type \"mv\" LinOp and override "
      "matvec() instead.")]] void
      set_elem(const cytnx_uint64 &i, const cytnx_uint64 &j, const T &elem,
               const bool check_exists = true) {
      this->_elems_it = this->_elems.find(i);
      if (this->_elems_it == this->_elems.end()) {
        // not exists:
        Tensor x({1}, this->_dtype);
        x(0) = elem;
        this->_elems[i] = std::pair<std::vector<cytnx_uint64>, Tensor>({j}, x);

      } else {
        std::vector<cytnx_uint64> &vi = this->_elems_it->second.first;  // pair:
        Tensor &ie = this->_elems_it->second.second;
        if (check_exists) {
          cytnx_error_msg(std::find(vi.begin(), vi.end(), j) != vi.end(),
                          "[ERROR] the element is set%s", "\n");
        }
        vi.push_back(j);
        ie.append(elem);
      }
    };
    /**
    @deprecated Part of the deprecated "mv_elem" path. Override matvec() with a type "mv" LinOp
    instead.
    */
    [[deprecated(
      "The \"mv_elem\" path (operator()(i, j)) is deprecated; construct a type \"mv\" LinOp and "
      "override matvec() instead.")]] Tensor::Tproxy
      operator()(const cytnx_uint64 &i, const cytnx_uint64 &j) {
      //[Note that this can only call by mv_elem]
      // if the element is not exists, it will create one.
      this->_elems_it = this->_elems.find(i);
      if (this->_elems_it == this->_elems.end()) {
        // not exists:
        Tensor x({1}, this->_dtype);
        x(0) = 0;
        this->_elems[i] = std::pair<std::vector<cytnx_uint64>, Tensor>({j}, x);
        return this->_elems[i].second(0);
      } else {
        std::vector<cytnx_uint64> &vi = this->_elems_it->second.first;  // pair:
        Tensor &ie = this->_elems_it->second.second;
        auto tmp_it = std::find(vi.begin(), vi.end(), j);

        // if(check_exists){
        //     cytnx_error_msg(std::find(vi.begin(), vi.end(), j)!=vi.end(),"[ERROR] the element is
        //     set%s","\n");
        // }
        if (tmp_it == vi.end()) {
          vi.push_back(j);
          ie.append(0);
          return ie(vi.size() - 1);
        } else {
          return ie(std::distance(vi.begin(), tmp_it));
        }
      }
    }

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

    void _print();

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
