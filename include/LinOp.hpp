#ifndef _H_LinOp_
#define _H_LinOp_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include "Scalar.hpp"
#include "UniTensor.hpp"
#include <vector>
#include <fstream>
#include <functional>
#include <map>
#include <utility>
#include <algorithm>
#include "intrusive_ptr_base.hpp"
#include "utils/vec_clone.hpp"

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
    @param type the type of operator, currently it can only be "mv" (matvec) or "mv_elem" (matvec
    with pre-store element)
    @param nx the last dimension of operator, this should be the dimension of the input vector when
    "mv_elem" is used.
    @param dtype the Operator's dtype. Note that this should match the input/output Tensor's dtype.
    @param device the Operator's on device.

    ## Note:
        1. the device and dtype should be set. This should be the same as the input and output
    vectors. by default, we assume custom_f take input and output vector to be on CPU and Double
    type.

    ## Details:
        The LinOp class is a class that defines a custom Linear operation acting on a Tensor or
    UniTensor. To use, inherit this class and override the matvec function. See the following
    examples for how to use them.

    ## Example:
    ### c++ API:
    \include example/LinOp/init.cpp
    #### output>
    \verbinclude example/LinOp/init.cpp.out
    ### python API:
    \include example/LinOp/init.py
    #### output>
    \verbinclude example/LinOp/init.py.out

    */
    LinOp(const std::string &type, const cytnx_uint64 &nx, const int &dtype = Type.Double,
          const int &device = Device.cpu) {
      if (type == "mv") {
      } else if (type == "mv_elem") {
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
            cytnx_error_msg(true,"[ERROR] cannot specify func with type=mv_elem%s. use set_elem
    instead.","\n");
        }
    };
    */
    template <class T>
    void set_elem(const cytnx_uint64 &i, const cytnx_uint64 &j, const T &elem,
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
    Tensor::Tproxy operator()(const cytnx_uint64 &i, const cytnx_uint64 &j) {
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

#endif
