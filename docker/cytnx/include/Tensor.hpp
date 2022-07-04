#ifndef _H_Tensor_
#define _H_Tensor_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Storage.hpp"
#include "Device.hpp"
#include "intrusive_ptr_base.hpp"
#include "utils/utils_internal_interface.hpp"
#include <iostream>
#include <vector>
#include <initializer_list>
#include "Accessor.hpp"
namespace cytnx {

  ///@cond
  // real implementation
  class Tensor_impl : public intrusive_ptr_base<Tensor_impl> {
   private:
    // Interface:
    Storage_init_interface __SII;

    // Memory:
    Storage _storage;

    // tensor shape
    std::vector<cytnx_uint64> _shape;

    // psudo-perm info
    std::vector<cytnx_uint64> _mapper;
    std::vector<cytnx_uint64> _invmapper;
    bool _contiguous;

   public:
    boost::intrusive_ptr<Tensor_impl> _clone_meta_only() const {
      boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());
      out->_mapper = this->_mapper;
      out->_invmapper = this->_invmapper;
      out->_shape = this->_shape;
      out->_contiguous = this->_contiguous;
      return out;
    }
    Tensor_impl() : _contiguous(true){};

    void Init(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype = Type.Double,
              const int device = -1);

    /*
    template<class T>
    void From_vec(const T &ndvec){
        cytnx_error_msg(std::string(typeid(T).name()).find("vector") ==
    std::string::npos,"[ERROR][Tensor][From_vec] the input argument should be a nd vector.%s","\n");
        //dispatch the rank!:



    }
    */
    // clone&assignment constr., use intrusive_ptr's
    Tensor_impl(const Tensor_impl &rhs);
    Tensor_impl &operator=(const Tensor_impl &rhs);  // add const

    unsigned int dtype() const { return this->_storage.dtype(); }
    int device() const { return this->_storage.device(); }

    std::string dtype_str() const { return Type.getname(this->_storage.dtype()); }
    std::string device_str() const { return Device.getname(this->_storage.device()); }

    const std::vector<cytnx_uint64> &shape() const { return _shape; }

    const bool &is_contiguous() const { return this->_contiguous; }

    const std::vector<cytnx_uint64> &mapper() const { return this->_mapper; }
    const std::vector<cytnx_uint64> &invmapper() const { return this->_invmapper; }
    Storage &storage() { return _storage; }

    const Storage &storage() const { return _storage; }

    boost::intrusive_ptr<Tensor_impl> clone() const {
      boost::intrusive_ptr<Tensor_impl> out = this->_clone_meta_only();
      out->_storage = this->_storage.clone();
      return out;
    }

    void to_(const int &device) { this->_storage.to_(device); }
    boost::intrusive_ptr<Tensor_impl> to(const int &device) {
      if (this->device() == device) {
        // boost::intrusive_ptr<Tensor_impl> out(this);
        return this;
      } else {
        boost::intrusive_ptr<Tensor_impl> out = this->_clone_meta_only();
        out->_storage = this->_storage.to(device);
        return out;
      }
    }

    void permute_(const std::vector<cytnx_uint64> &rnks);

    boost::intrusive_ptr<Tensor_impl> permute(const std::vector<cytnx_uint64> &rnks) {
      boost::intrusive_ptr<Tensor_impl> out = this->clone();
      out->permute_(rnks);
      return out;
    }

    template <class T>
    T &at(const std::vector<cytnx_uint64> &locator) {
      cytnx_error_msg(locator.size() != this->_shape.size(), "%s",
                      "The input indexes rank is not match Tensor's rank.");

      cytnx_uint64 RealRank, mtplyr;
      std::vector<cytnx_uint64> c_shape(this->_shape.size());
      std::vector<cytnx_uint64> c_loc(this->_shape.size());

      RealRank = 0;
      mtplyr = 1;

      for (cytnx_int64 i = this->_shape.size() - 1; i >= 0; i--) {
        if (locator[i] >= this->_shape[i]) {
          cytnx_error_msg(
            true, "%s",
            "The dimension of rank that trying to access is exceed Tensor's dimension.");
        }
        c_shape[i] = this->_shape[this->_invmapper[i]];
        c_loc[i] = locator[this->_invmapper[i]];
        RealRank += mtplyr * c_loc[i];
        mtplyr *= c_shape[i];
      }
      return this->_storage.at<T>(RealRank);
    }

    boost::intrusive_ptr<Tensor_impl> get(const std::vector<cytnx::Accessor> &accessors);
    void set(const std::vector<cytnx::Accessor> &accessors,
             const boost::intrusive_ptr<Tensor_impl> &rhs);

    template <class T>
    void set(const std::vector<cytnx::Accessor> &accessors, const T &rc);

    template <class Tx>
    void fill(const Tx &val) {
      this->storage().fill(val);
    }

    boost::intrusive_ptr<Tensor_impl> contiguous() {
      // return new instance if act on non-contiguous tensor
      // return self if act on contiguous tensor
      if (this->_contiguous) {
        boost::intrusive_ptr<Tensor_impl> out(this);
        return out;
      } else {
        boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());
        std::vector<cytnx_uint64> oldshape(this->_shape.size());
        for (cytnx_uint64 i = 0; i < this->_shape.size(); i++) {
          oldshape[i] = this->_shape[this->_invmapper[i]];
        }

        out->_storage =
          this->_storage._impl->Move_memory(oldshape, this->_mapper, this->_invmapper);
        out->_invmapper = utils_internal::range_cpu(this->_invmapper.size());
        out->_mapper = out->_invmapper;
        out->_shape = this->_shape;
        out->_contiguous = true;
        return out;
      }
    }

    void contiguous_() {
      // return new instance if act on non-contiguous tensor
      // return self if act on contiguous tensor
      if (!this->_contiguous) {
        std::vector<cytnx_uint64> oldshape(this->_shape.size());
        for (cytnx_uint64 i = 0; i < this->_shape.size(); i++) {
          oldshape[i] = this->_shape[this->_invmapper[i]];
        }
        this->_storage._impl->Move_memory_(oldshape, this->_mapper, this->_invmapper);
        this->_mapper = utils_internal::range_cpu(this->_invmapper.size());
        this->_invmapper = this->_mapper;
        this->_contiguous = true;
      }
    }

    void reshape_(const std::vector<cytnx_int64> &new_shape) {
      if (!this->_contiguous) {
        this->contiguous_();
      }
      std::vector<cytnx_uint64> result_shape(new_shape.size());
      cytnx_uint64 new_N = 1;
      bool has_undetermine = false;
      unsigned int Udet_id = 0;
      for (int i = 0; i < new_shape.size(); i++) {
        if (new_shape[i] < 0) {
          if (new_shape[i] != -1)
            cytnx_error_msg(
              new_shape[i] != -1, "%s",
              "[ERROR] reshape can only have dimension > 0 and one undetermine rank specify as -1");
          if (has_undetermine)
            cytnx_error_msg(
              new_shape[i] != -1, "%s",
              "[ERROR] reshape can only have dimension > 0 and one undetermine rank specify as -1");
          Udet_id = i;
          has_undetermine = true;
        } else {
          new_N *= new_shape[i];
          result_shape[i] = new_shape[i];
        }
      }

      if (has_undetermine) {
        cytnx_error_msg(new_N >= this->_storage.size(), "%s",
                        "[ERROR] new shape exceed the total number of elements.");
        cytnx_error_msg(this->_storage.size() % new_N, "%s",
                        "[ERROR] unmatch size when reshape with undetermine dimension");
        result_shape[Udet_id] = this->_storage.size() / new_N;
      } else {
        cytnx_error_msg(new_N != this->_storage.size(), "%s",
                        "[ERROR] new shape does not match the number of elements.");
      }

      this->_shape = result_shape;
      this->_mapper = utils_internal::range_cpu(result_shape.size());
      this->_invmapper = this->_mapper;
    }

    boost::intrusive_ptr<Tensor_impl> reshape(const std::vector<cytnx_int64> &new_shape) {
      boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());
      if (!this->_contiguous) {
        out = this->contiguous();
      } else {
        out = this->clone();
      }

      out->reshape_(new_shape);
      return out;
    }

    boost::intrusive_ptr<Tensor_impl> astype(const int &new_type) {
      // boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());
      // out->_storage = this->_storage.astype(new_type);
      if (this->dtype() == new_type) {
        return this;
      } else {
        boost::intrusive_ptr<Tensor_impl> out = this->_clone_meta_only();
        out->_storage = this->_storage.astype(new_type);
        return out;
      }
    }
  };
  ///@endcond

  /// @brief an tensor (multi-dimensional array)
  class Tensor {
   private:
   public:
    ///@cond
    boost::intrusive_ptr<Tensor_impl> _impl;
    Tensor() : _impl(new Tensor_impl()){};
    Tensor(const Tensor &rhs) { _impl = rhs._impl; }
    Tensor &operator=(const Tensor &rhs) { _impl = rhs._impl; }
    ///@endcond

    //@{
    // default device==Device.cpu (-1)
    /**
    @brief initialize a Tensor
    @param shape the shape of tensor.
    @param dtype the dtype of tensor. This can be any of type defined in cytnx::Type
    @param device the device that tensor to be created. This can be cytnx::Device.cpu or
    cytnx::Device.cuda+<gpuid>

    [Note]
        1. the content of Tensor created will be un-initialize! See \link cytnx::zeros
    zeros()\endlink, \link cytnx::ones ones() \endlink or \link cytnx::arange arange() \endlink for
    generating an Tensor.

    ## Example:
    ### c++ API:
    \include example/Tensor/Init.cpp
    #### output>
    \verbinclude example/Tensor/Init.cpp.out
    ### python API:
    \include example/Tensor/Init.py
    #### output>
    \verbinclude example/Tensor/Init.py.out
    */
    void Init(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype = Type.Double,
              const int &device = -1) {
      boost::intrusive_ptr<Tensor_impl> tmp(new Tensor_impl());
      this->_impl = tmp;
      this->_impl->Init(shape, dtype, device);
    }

    Tensor(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype = Type.Double,
           const int &device = -1)
        : _impl(new Tensor_impl()) {
      this->Init(shape, dtype, device);
    }
    //@}

    /**
    @brief the dtype-id of the Tensor
    @return [cytnx_uint64] the dtype_id of the Tensor

    */
    unsigned int dtype() const { return this->_impl->dtype(); }

    /**
    @brief the device-id of the Tensor
    @return [cytnx_int64] the device_id of the Tensor

    */
    int device() const { return this->_impl->device(); }

    /**
    @brief the dtype (in string) of the Tensor
    @return [std::string] the dtype of the Tensor

    */
    std::string dtype_str() const { return this->_impl->dtype_str(); }

    /**
    @brief the device (in string) of the Tensor
    @return [std::string] the device of the Tensor

    */
    std::string device_str() const { return this->_impl->device_str(); }

    /**
    @brief the shape of the Tensor
    @return [std::vector<cytnx_uint64>] the shape of the Tensor

    */
    const std::vector<cytnx_uint64> &shape() const { return this->_impl->shape(); }

    /**
    @brief return a clone of the current Tensor.
    @return [Tensor]

    description:\n
        In C++ API, the behavior of assignment operator is designed to have same behavior as
    python,\n to have a copy of the current tensor, we call clone to return a copy.

    ## Example:
    ### c++ API:
    \include example/Tensor/clone.cpp
    #### output>
    \verbinclude example/Tensor/clone.cpp.out
    ### python API:
    \include example/Tensor/clone.py
    #### output>
    \verbinclude example/Tensor/clone.py.out
    */
    Tensor clone() const {
      Tensor out;
      out._impl = this->_impl->clone();
      return out;
    }

    /**
    @brief copy a tensor to new device
    @param device the device-id that is moving to. it can be any device defined in cytnx::Device
    @return [Tensor]

    description:\n
        if the device-id is the same as current Tensor's device, then return self.\n
        otherwise, return a copy of instance that located on the target device.   \n
        see also: \link cytnx::Tensor::to_ Tensor.to_ \endlink                    \n

    ## Example:
    ### c++ API:
    \include example/Tensor/to.cpp
    #### output>
    \verbinclude example/Tensor/to.cpp.out
    ### python API:
    \include example/Tensor/to.py
    #### output>
    \verbinclude example/Tensor/to.py.out
    */
    Tensor to(const int &device) const {
      Tensor out;
      out._impl = this->_impl->to(device);
      return out;
    }

    /**
    @brief move the current Tensor to the device.
    @param device the device-id that is moving to. it can be any device defined in cytnx::Device

    description:\n
        see also: \link cytnx::Tensor::to Tensor.to \endlink\n

    ## Example:
    ### c++ API:
    \include example/Tensor/to_.cpp
    #### output>
    \verbinclude example/Tensor/to_.cpp.out
    ### python API:
    \include example/Tensor/to_.py
    #### output>
    \verbinclude example/Tensor/to_.py.out
    */
    void to_(const int &device) { this->_impl->to_(device); }

    const bool &is_contiguous() const { return this->_impl->is_contiguous(); }

    void permute_(const std::vector<cytnx_uint64> &rnks) { this->_impl->permute_(rnks); }

    /**
    @brief perform tensor permute on the cytnx::Tensor and return a new instance.
    @param rnks the permute indices, should have No. of elements equal to the rank of tensor.
    @return [Tensor] a permuted new Tensor

    ## Example:
    ### c++ API:
    \include example/Tensor/permute.cpp
    #### output>
    \verbinclude example/Tensor/permute.cpp.out
    ### python API:
    \include example/Tensor/permute.py
    #### output>
    \verbinclude example/Tensor/permute.py.out
    */
    Tensor permute(const std::vector<cytnx_uint64> &rnks) const {
      Tensor out;
      out._impl = this->_impl->permute(rnks);
      return out;
    }

    /**
    @brief Make the Tensor contiguous by coalescing the memory (storage).
    @return [Tensor] a new Tensor that is with contiguous memory (storage).

    See also \link Tensor::contiguous_ Tensor::contiguous_() \endlink

    ## Example:
    ### c++ API:
    \include example/Tensor/contiguous.cpp
    #### output>
    \verbinclude example/Tensor/contiguous.cpp.out
    ### python API:
    \include example/Tensor/contiguous.py
    #### output>
    \verbinclude example/Tensor/contiguous.py.out
    */
    Tensor contiguous() {
      Tensor out;
      out._impl = this->_impl->contiguous();
      return out;
    }

    /**
    @brief Make the Tensor contiguous by coalescing the memory (storage), inplacely

    See also \link Tensor::contiguous Tensor::contiguous() \endlink

    ## Example:
    ### c++ API:
    \include example/Tensor/contiguous_.cpp
    #### output>
    \verbinclude example/Tensor/contiguous_.cpp.out
    ### python API:
    \include example/Tensor/contiguous_.py
    #### output>
    \verbinclude example/Tensor/contiguous_.py.out
    */
    void contiguous_() { this->_impl->contiguous_(); }

    /**
    @brief reshape the Tensor, inplacely
    @param new_shape the new shape of the Tensor.

    See also \link Tensor::reshape Tensor::reshape() \endlink

    ## Example:
    ### c++ API:
    \include example/Tensor/reshape_.cpp
    #### output>
    \verbinclude example/Tensor/reshape_.cpp.out
    ### python API:
    \include example/Tensor/reshape_.py
    #### output>
    \verbinclude example/Tensor/reshape_.py.out
    */
    void reshape_(const std::vector<cytnx_int64> &new_shape) { this->_impl->reshape_(new_shape); }

    /**
    @brief return a new Tensor that is reshaped.
    @param new_shape the new shape of the Tensor.
    @return [Tensor]

    See also \link Tensor::reshape_ Tensor::reshape_() \endlink

    ## Example:
    ### c++ API:
    \include example/Tensor/reshape.cpp
    #### output>
    \verbinclude example/Tensor/reshape.cpp.out
    ### python API:
    \include example/Tensor/reshape.py
    #### output>
    \verbinclude example/Tensor/reshape.py.out
    */
    Tensor reshape(const std::vector<cytnx_int64> &new_shape) {
      Tensor out;
      out._impl = this->_impl->reshape(new_shape);
      return out;
    }

    /**
    @brief return a new Tensor that cast to different dtype.
    @param new_type the new dtype. It can be any type defined in cytnx::Type
    @return [Tensor]

    ## Note:
        If the new_type is the same as dtype of the current Tensor, return self.

    ## Example:
    ### c++ API:
    \include example/Tensor/astype.cpp
    #### output>
    \verbinclude example/Tensor/astype.cpp.out
    ### python API:
    \include example/Tensor/astype.py
    #### output>
    \verbinclude example/Tensor/astype.py.out
    */
    Tensor astype(const int &new_type) const {
      Tensor out;
      out._impl = this->_impl->astype(new_type);
      return out;
    }

    /**
    @brief [C++ only] get an element at specific location.
    @param locator the location of the element
    @return [ref]

    ## Note:
        1. This is for C++ API only!
        2. need template instantiation to resolve the type, which should be consist with the dtype
    of the Tensor. An error will be issued if the template type is inconsist with the current dtype
    of Tensor.
        3. For python API, use [] directly to get element.

    ## Example:
    ### c++ API:
    \include example/Tensor/at.cpp
    #### output>
    \verbinclude example/Tensor/at.cpp.out
    */
    template <class T>
    T &at(const std::vector<cytnx_uint64> &locator) {
      return this->_impl->at<T>(locator);
    }

    /**
    @brief get an from a rank-0 Tensor
    @return [T]

    ## Note:
        1. This can only be called on a rank-0 Tensor (scalar). For C++ API, a template
    instantiation of type is needed to resolve the type, which should be connsist with the dtype of
    the Tensor. An error will be issued if the template type if inconsist with the current dtype of
    Tensor.
        2. Although the return is by reference in C++ part, the return in python is not.
        3. From 2., We recommend user to use at<T> (C++ API) and [] (python API) to modify the value
    of the element to have consistant syntax across two languages.

    ## Example:
    ### c++ API:
    \include example/Tensor/item.cpp
    #### output>
    \verbinclude example/Tensor/item.cpp.out
    ### python API:
    \include example/Tensor/item.py
    #### output>
    \verbinclude example/Tensor/item.py.out
    */
    template <class T>
    T &item() {
      cytnx_error_msg(this->_impl->storage().size() != 1, "[ERROR][Tensor.item<T>]%s",
                      "item can only be called from a Tensor with only one element\n");
      return this->_impl->storage().at<T>(0);
    }

    /**
    @brief get elements using Accessor (C++ API) / slices (python API)
    @return [Tensor]

    See also \link cytnx::Accessor Accessor\endlink for cordinate with Accessor in C++ API.

    ## Note:
        1. the return will be a new Tensor instance, which not share memory with the current Tensor.


    ## Example:
    ### c++ API:
    \include example/Tensor/get.cpp
    #### output>
    \verbinclude example/Tensor/get.cpp.out
    ### python API:
    \include example/Tensor/get.py
    #### output>
    \verbinclude example/Tensor/get.py.out
    */
    Tensor get(const std::vector<cytnx::Accessor> &accessors) const {
      Tensor out;
      out._impl = this->_impl->get(accessors);
      return out;
    }

    /**
    @brief set elements with the input Tensor using Accessor (C++ API) / slices (python API)
    @param accessors the list(vector) of accessors.
    @param rhs [Tensor]


    ## Note:
        the shape of the input Tensor should be the same as the shape that indicated using Accessor.
    The memory is not shared with the input Tensor.

    ## Example:
    ### c++ API:
    \include example/Tensor/set.cpp
    #### output>
    \verbinclude example/Tensor/set.cpp.out
    ### python API:
    \include example/Tensor/set.py
    #### output>
    \verbinclude example/Tensor/set.py.out
    */
    void set(const std::vector<cytnx::Accessor> &accessors, const Tensor &rhs) {
      this->_impl->set(accessors, rhs._impl);
    }

    /**
    @brief set elements with the input constant using Accessor (C++ API) / slices (python API)
    @param accessors the list(vector) of accessors.
    @param rc [Const]

    See also \link cytnx::Tensor::fill Tensor::fill \endlink for filling all elements with assigned
    constant.

    ## Example:
    ### c++ API:
    \include example/Tensor/set.cpp
    #### output>
    \verbinclude example/Tensor/set.cpp.out
    ### python API:
    \include example/Tensor/set.py
    #### output>
    \verbinclude example/Tensor/set.py.out
    */
    template <class T>
    void set(const std::vector<cytnx::Accessor> &accessors, const T &rc) {
      this->_impl->set(accessors, rc);
    }
    ///@cond
    template <class T>
    void set(const std::initializer_list<cytnx::Accessor> &accessors, const T &rc) {
      std::vector<cytnx::Accessor> args = accessors;
      this->set(args, rc);
    }
    ///@endcond

    /**
    @brief return the storage of current Tensor.
    @return [Storage]

    ## Note:
        1. The return storage shares the same instance of the storage of current Tensor. Use \link
    Storage::clone Storage.clone() \endlink to create a new instance of the returned Storage.

    */
    Storage &storage() const { return this->_impl->storage(); }

    /**
    @brief fill all the element of current Tensor with the value.
    @param val the assigned value


    ## Example:
    ### c++ API:
    \include example/Tensor/fill.cpp
    #### output>
    \verbinclude example/Tensor/fill.cpp.out
    ### python API
    \include example/Tensor/fill.py
    #### output>
    \verbinclude example/Tensor/fill.py.out
    */
    template <class T>
    void fill(const T &val) {
      this->_impl->fill(val);
    }

    bool equiv(const Tensor &rhs) {
      if (this->shape() != rhs.shape()) return false;
      return true;
    }

    // Arithmic:
    template <class T>
    Tensor &operator+=(const T &rc);
    template <class T>
    Tensor &operator-=(const T &rc);
    template <class T>
    Tensor &operator*=(const T &rc);
    template <class T>
    Tensor &operator/=(const T &rc);

    template <class T>
    Tensor Add(const T &rhs) {
      return *this + rhs;
    }
    template <class T>
    Tensor &Add_(const T &rhs) {
      return *this += rhs;
    }

    template <class T>
    Tensor Sub(const T &rhs) {
      return *this - rhs;
    }
    template <class T>
    Tensor &Sub_(const T &rhs) {
      return *this -= rhs;
    }

    template <class T>
    Tensor Mul(const T &rhs) {
      return *this * rhs;
    }
    template <class T>
    Tensor &Mul_(const T &rhs) {
      return *this *= rhs;
    }

    template <class T>
    Tensor Div(const T &rhs) {
      return *this / rhs;
    }
    template <class T>
    Tensor &Div_(const T &rhs) {
      return *this /= rhs;
    }

    template <class T>
    Tensor Cpr(const T &rhs) {
      return *this == rhs;
    }
    template <class T>
    Tensor &Cpr_(const T &rhs) {
      return *this == rhs;
    }

    // linalg:
    std::vector<Tensor> Svd(const bool &is_U = true, const bool &is_vT = true);
    std::vector<Tensor> Eigh(const bool &is_V = false);
    Tensor &Inv_();
    Tensor Inv();
    Tensor &Conj_();
    Tensor Conj();
    Tensor &Exp_();
    Tensor Exp();

  };  // class Tensor

  std::ostream &operator<<(std::ostream &os, const Tensor &in);

}  // namespace cytnx

#endif
