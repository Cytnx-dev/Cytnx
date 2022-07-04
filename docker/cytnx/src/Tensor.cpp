#include <typeinfo>
#include "Tensor.hpp"
#include "utils/utils_internal_interface.hpp"
#include "linalg/linalg.hpp"

using namespace std;

namespace cytnx {

  void Tensor_impl::Init(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype,
                         int device) {
    // check:
    cytnx_error_msg(dtype >= N_Type, "%s", "[ERROR] invalid argument: dtype");
    cytnx_error_msg(shape.size() == 0, "%s",
                    "[ERROR] invalid argument: shape. Must at least have one element.");
    cytnx_uint64 Nelem = 1;
    for (int i = 0; i < shape.size(); i++) {
      cytnx_error_msg(shape[i] == 0, "%s", "[ERROR] shape cannot have 0 dimension in any rank.");
      Nelem *= shape[i];
    }
    // this->_storage = __SII.USIInit[dtype]();
    this->_storage.Init(Nelem, dtype, device);
    this->_shape = shape;
    this->_mapper = utils_internal::range_cpu(shape.size());
    this->_invmapper = this->_mapper;
    this->_contiguous = true;
  }

  void Tensor_impl::permute_(const std::vector<cytnx_uint64> &rnks) {
    // check::
    if (rnks.size() != this->_shape.size()) {
      cytnx_error_msg(true, "%s",
                      "reshape a tensor with a specify shape that does not match with the shape of "
                      "the incident tensor.");
    }

    std::vector<cytnx_uint64> new_fwdmap(this->_shape.size());
    std::vector<cytnx_uint64> new_shape(this->_shape.size());
    std::vector<cytnx_uint64> new_idxmap(this->_shape.size());

    // for(int i=0;i<this->_shape.size();i++)
    //     std::cout << this->_mapper[i] << " " << this->_invmapper[i] << std::endl;

    for (cytnx_uint32 i = 0; i < rnks.size(); i++) {
      if (rnks[i] >= rnks.size()) {
        cytnx_error_msg(1, "%s", "reshape a tensor with invalid rank index.");
      }
      // std::cout << this->_mapper[rnks[i]] << " " << i << std::endl;
      new_idxmap[this->_mapper[rnks[i]]] = i;
      new_fwdmap[i] = this->_mapper[rnks[i]];
      new_shape[i] = this->_shape[rnks[i]];
    }

    this->_invmapper = new_idxmap;
    this->_shape = new_shape;
    this->_mapper = new_fwdmap;

    /// checking if permute back to contiguous:
    bool iconti = true;
    for (cytnx_uint32 i = 0; i < rnks.size(); i++) {
      if (new_fwdmap[i] != new_idxmap[i]) {
        iconti = false;
        break;
      }
      if (new_fwdmap[i] != i) {
        iconti = false;
        break;
      }
    }
    this->_contiguous = iconti;
  }

  boost::intrusive_ptr<Tensor_impl> Tensor_impl::get(
    const std::vector<cytnx::Accessor> &accessors) {
    cytnx_error_msg(accessors.size() != this->_shape.size(), "%s",
                    "The input indexes rank is not match Tensor's rank.");

    vector<cytnx_uint64> get_shape(accessors.size());
    // vector<cytnx_uint64> new_shape;
    std::vector<std::vector<cytnx_uint64>> locators(accessors.size());
    for (cytnx_uint32 i = 0; i < accessors.size(); i++) {
      accessors[i].get_len_pos(this->_shape[i], get_shape[i], locators[i]);
      // std::cout << this->_shape[i] << " " << get_shape[i] << "|";
      // for(int j=0;j<locators[i].size();j++) std::cout << locators[i][j] << " ";
      // std::cout << std::endl;
    }

    boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());
    out->Init(get_shape, this->dtype(), this->device());

    this->storage()._impl->GetElem_byShape(out->storage()._impl, this->shape(), this->_mapper,
                                           get_shape, locators);

    vector<cytnx_int64> new_shape;
    for (cytnx_uint32 i = 0; i < accessors.size(); i++)
      if (get_shape[i] != 1) new_shape.push_back(get_shape[i]);

    if (new_shape.size() == 0)
      out->reshape_({1});
    else
      out->reshape_(new_shape);
    return out;
  }

  void Tensor_impl::set(const std::vector<cytnx::Accessor> &accessors,
                        const boost::intrusive_ptr<Tensor_impl> &rhs) {
    cytnx_error_msg(accessors.size() != this->_shape.size(), "%s",
                    "The input indexes rank is not match Tensor's rank.");

    vector<cytnx_uint64> get_shape(accessors.size());
    // vector<cytnx_uint64> new_shape;

    std::vector<std::vector<cytnx_uint64>> locators(accessors.size());
    for (cytnx_uint32 i = 0; i < accessors.size(); i++) {
      accessors[i].get_len_pos(this->_shape[i], get_shape[i], locators[i]);
      // std::cout << this->_shape[i] << " " << get_shape[i] << "|";
      // for(int j=0;j<locators[i].size();j++) std::cout << locators[i][j] << " ";
      // std::cout << std::endl;
    }

    // remove single dim
    vector<cytnx_uint64> new_shape;
    for (cytnx_uint32 i = 0; i < accessors.size(); i++)
      if (get_shape[i] != 1) new_shape.push_back(get_shape[i]);

    if (new_shape.size() == 0) new_shape.push_back(1);

    // check size:
    cytnx_error_msg(new_shape != rhs->shape(), "[ERROR][Tensor.set_elems]%s", "inconsistent shape");

    // boost::intrusive_ptr<Tensor_impl> out( new Tensor_impl());
    // out->Init(get_shape,this->dtype(),this->device());

    this->storage()._impl->SetElem_byShape(rhs->storage()._impl, this->shape(), this->_mapper,
                                           get_shape, locators, false);
  }

  template <class T>
  void Tensor_impl::set(const std::vector<cytnx::Accessor> &accessors, const T &rc) {
    cytnx_error_msg(accessors.size() != this->_shape.size(), "%s",
                    "The input indexes rank is not match Tensor's rank.");

    vector<cytnx_uint64> get_shape(accessors.size());
    // vector<cytnx_uint64> new_shape;

    std::vector<std::vector<cytnx_uint64>> locators(accessors.size());
    for (cytnx_uint32 i = 0; i < accessors.size(); i++) {
      accessors[i].get_len_pos(this->_shape[i], get_shape[i], locators[i]);
      // std::cout << this->_shape[i] << " " << get_shape[i] << "|";
      // for(int j=0;j<locators[i].size();j++) std::cout << locators[i][j] << " ";
      // std::cout << std::endl;
    }

    // remove single dim
    vector<cytnx_uint64> new_shape;
    for (cytnx_uint32 i = 0; i < accessors.size(); i++)
      if (get_shape[i] != 1) new_shape.push_back(get_shape[i]);

    if (new_shape.size() == 0) new_shape.push_back(1);

    // boost::intrusive_ptr<Tensor_impl> out( new Tensor_impl());
    // out->Init(get_shape,this->dtype(),this->device());

    Storage tmp(1, Type.c_typename_to_id(typeid(T).name()), this->device());
    tmp.at<T>(0) = rc;
    this->storage()._impl->SetElem_byShape(tmp._impl, this->shape(), this->_mapper, get_shape,
                                           locators, true);
  }
  template void Tensor_impl::set<cytnx_complex128>(const std::vector<cytnx::Accessor> &,
                                                   const cytnx_complex128 &);
  template void Tensor_impl::set<cytnx_complex64>(const std::vector<cytnx::Accessor> &,
                                                  const cytnx_complex64 &);
  template void Tensor_impl::set<cytnx_double>(const std::vector<cytnx::Accessor> &,
                                               const cytnx_double &);
  template void Tensor_impl::set<cytnx_float>(const std::vector<cytnx::Accessor> &,
                                              const cytnx_float &);
  template void Tensor_impl::set<cytnx_int64>(const std::vector<cytnx::Accessor> &,
                                              const cytnx_int64 &);
  template void Tensor_impl::set<cytnx_uint64>(const std::vector<cytnx::Accessor> &,
                                               const cytnx_uint64 &);
  template void Tensor_impl::set<cytnx_int32>(const std::vector<cytnx::Accessor> &,
                                              const cytnx_int32 &);
  template void Tensor_impl::set<cytnx_uint32>(const std::vector<cytnx::Accessor> &,
                                               const cytnx_uint32 &);
  template void Tensor_impl::set<cytnx_int16>(const std::vector<cytnx::Accessor> &,
                                              const cytnx_int16 &);
  template void Tensor_impl::set<cytnx_uint16>(const std::vector<cytnx::Accessor> &,
                                               const cytnx_uint16 &);
  template void Tensor_impl::set<cytnx_bool>(const std::vector<cytnx::Accessor> &,
                                             const cytnx_bool &);

  std::ostream &operator<<(std::ostream &os, const Tensor &in) {
    if (in.is_contiguous())
      in._impl->storage()._impl->PrintElem_byShape(os, in.shape());
    else
      in._impl->storage()._impl->PrintElem_byShape(os, in.shape(), in._impl->invmapper());
    return os;
  }

  //===================================================================
  // wrapper

  // +=
  template <>
  Tensor &Tensor::operator+=<Tensor>(const Tensor &rc) {
    *this = cytnx::linalg::Add(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_complex128>(const cytnx_complex128 &rc) {
    *this = cytnx::linalg::Add(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_complex64>(const cytnx_complex64 &rc) {
    *this = cytnx::linalg::Add(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_double>(const cytnx_double &rc) {
    *this = cytnx::linalg::Add(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_float>(const cytnx_float &rc) {
    *this = cytnx::linalg::Add(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_int64>(const cytnx_int64 &rc) {
    *this = cytnx::linalg::Add(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_uint64>(const cytnx_uint64 &rc) {
    *this = cytnx::linalg::Add(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_int32>(const cytnx_int32 &rc) {
    *this = cytnx::linalg::Add(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_uint32>(const cytnx_uint32 &rc) {
    *this = cytnx::linalg::Add(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_int16>(const cytnx_int16 &rc) {
    *this = cytnx::linalg::Add(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_uint16>(const cytnx_uint16 &rc) {
    *this = cytnx::linalg::Add(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_bool>(const cytnx_bool &rc) {
    *this = cytnx::linalg::Add(*this, rc);
    return *this;
  }

  // -=
  template <>
  Tensor &Tensor::operator-=<Tensor>(const Tensor &rc) {
    *this = cytnx::linalg::Sub(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_complex128>(const cytnx_complex128 &rc) {
    *this = cytnx::linalg::Sub(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_complex64>(const cytnx_complex64 &rc) {
    *this = cytnx::linalg::Sub(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_double>(const cytnx_double &rc) {
    *this = cytnx::linalg::Sub(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_float>(const cytnx_float &rc) {
    *this = cytnx::linalg::Sub(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_int64>(const cytnx_int64 &rc) {
    *this = cytnx::linalg::Sub(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_uint64>(const cytnx_uint64 &rc) {
    *this = cytnx::linalg::Sub(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_int32>(const cytnx_int32 &rc) {
    *this = cytnx::linalg::Sub(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_uint32>(const cytnx_uint32 &rc) {
    *this = cytnx::linalg::Sub(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_int16>(const cytnx_int16 &rc) {
    *this = cytnx::linalg::Sub(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_uint16>(const cytnx_uint16 &rc) {
    *this = cytnx::linalg::Sub(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_bool>(const cytnx_bool &rc) {
    *this = cytnx::linalg::Sub(*this, rc);
    return *this;
  }

  // *=
  template <>
  Tensor &Tensor::operator*=<Tensor>(const Tensor &rc) {
    *this = cytnx::linalg::Mul(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_complex128>(const cytnx_complex128 &rc) {
    *this = cytnx::linalg::Mul(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_complex64>(const cytnx_complex64 &rc) {
    *this = cytnx::linalg::Mul(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_double>(const cytnx_double &rc) {
    *this = cytnx::linalg::Mul(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_float>(const cytnx_float &rc) {
    *this = cytnx::linalg::Mul(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_int64>(const cytnx_int64 &rc) {
    *this = cytnx::linalg::Mul(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_uint64>(const cytnx_uint64 &rc) {
    *this = cytnx::linalg::Mul(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_int32>(const cytnx_int32 &rc) {
    *this = cytnx::linalg::Mul(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_uint32>(const cytnx_uint32 &rc) {
    *this = cytnx::linalg::Mul(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_int16>(const cytnx_int16 &rc) {
    *this = cytnx::linalg::Mul(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_uint16>(const cytnx_uint16 &rc) {
    *this = cytnx::linalg::Mul(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_bool>(const cytnx_bool &rc) {
    *this = cytnx::linalg::Mul(*this, rc);
    return *this;
  }

  // /=
  template <>
  Tensor &Tensor::operator/=<Tensor>(const Tensor &rc) {
    *this = cytnx::linalg::Div(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_complex128>(const cytnx_complex128 &rc) {
    *this = cytnx::linalg::Div(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_complex64>(const cytnx_complex64 &rc) {
    *this = cytnx::linalg::Div(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_double>(const cytnx_double &rc) {
    *this = cytnx::linalg::Div(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_float>(const cytnx_float &rc) {
    *this = cytnx::linalg::Div(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_int64>(const cytnx_int64 &rc) {
    *this = cytnx::linalg::Div(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_uint64>(const cytnx_uint64 &rc) {
    *this = cytnx::linalg::Div(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_int32>(const cytnx_int32 &rc) {
    *this = cytnx::linalg::Div(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_uint32>(const cytnx_uint32 &rc) {
    *this = cytnx::linalg::Div(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_int16>(const cytnx_int16 &rc) {
    *this = cytnx::linalg::Div(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_uint16>(const cytnx_uint16 &rc) {
    *this = cytnx::linalg::Div(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_bool>(const cytnx_bool &rc) {
    *this = cytnx::linalg::Div(*this, rc);
    return *this;
  }

  std::vector<Tensor> Tensor::Svd(const bool &is_U, const bool &is_vT) {
    return linalg::Svd(*this, is_U, is_vT);
  }
  std::vector<Tensor> Tensor::Eigh(const bool &is_V) { return linalg::Eigh(*this, is_V); }

  Tensor &Tensor::Inv_() {
    linalg::Inv_(*this);
    return *this;
  }
  Tensor Tensor::Inv() { return linalg::Inv(*this); }

  Tensor &Tensor::Conj_() {
    linalg::Conj_(*this);
    return *this;
  }
  Tensor Tensor::Conj() { return linalg::Conj(*this); }

  Tensor &Tensor::Exp_() {
    linalg::Exp_(*this);
    return *this;
  }
  Tensor Tensor::Exp() { return linalg::Exp(*this); }

}  // namespace cytnx
