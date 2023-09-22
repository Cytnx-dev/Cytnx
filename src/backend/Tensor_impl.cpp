#include <typeinfo>
#include "backend/Tensor_impl.hpp"
#include "utils_internal_interface.hpp"
#include "linalg.hpp"
#include "utils/is.hpp"
#include "Type.hpp"
using namespace std;

namespace cytnx {

  //-----------------------------------------------
  void Tensor_impl::Init(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype,
                         int device, const bool &init_zero) {
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
    this->_storage.Init(Nelem, dtype, device, init_zero);
    this->_shape = shape;
    this->_mapper = vec_range(shape.size());
    this->_invmapper = this->_mapper;
    this->_contiguous = true;
    // cout << shape << endl;
  }
  void Tensor_impl::Init(const Storage &in) {
    cytnx_error_msg(in.dtype() == Type.Void,
                    "[ERROR] cannot init Tensor using un-initialized Storage%s", "\n");
    this->_storage = in;
    this->_shape.clear();
    this->_shape.push_back(in.size());
    this->_mapper.clear();
    this->_mapper.push_back(0);
    this->_invmapper = this->_mapper;
    this->_contiguous = true;
  }

  boost::intrusive_ptr<Tensor_impl> Tensor_impl::permute(const std::vector<cytnx_uint64> &rnks) {
    // check::
    if (rnks.size() != this->_shape.size()) {
      cytnx_error_msg(true, "%s",
                      "reshape a tensor with a specify shape that does not match with the shape of "
                      "the incident tensor.");
    }

    if (vec_unique(rnks).size() != rnks.size()) {
      cytnx_error_msg(true, "%s", "tensor permute with duplicated index.\n");
    }

    std::vector<cytnx_uint64> new_fwdmap(this->_shape.size());
    std::vector<cytnx_uint64> new_shape(this->_shape.size());
    std::vector<cytnx_uint64> new_idxmap(this->_shape.size());

    // for(int i=0;i<this->_shape.size();i++)
    //     std::cout << this->_mapper[i] << " " << this->_invmapper[i] << std::endl;

    boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());

    for (cytnx_uint32 i = 0; i < rnks.size(); i++) {
      if (rnks[i] >= rnks.size()) {
        cytnx_error_msg(1, "%s", "reshape a tensor with invalid rank index.");
      }
      // std::cout << this->_mapper[rnks[i]] << " " << i << std::endl;
      new_idxmap[this->_mapper[rnks[i]]] = i;
      new_fwdmap[i] = this->_mapper[rnks[i]];
      new_shape[i] = this->_shape[rnks[i]];
    }

    out->_invmapper = std::move(new_idxmap);
    out->_shape = std::move(new_shape);
    out->_mapper = std::move(new_fwdmap);

    /// checking if permute back to contiguous:
    bool iconti = true;
    for (cytnx_uint32 i = 0; i < rnks.size(); i++) {
      // if (new_fwdmap[i] != new_idxmap[i]) {
      //   iconti = false;
      //   break;
      // }
      if (out->_mapper[i] != i) {
        iconti = false;
        break;
      }
    }
    out->_contiguous = iconti;

    // ref storage
    out->_storage = this->_storage;
    return out;
  }

  void Tensor_impl::permute_(const std::vector<cytnx_uint64> &rnks) {
    // check::
    if (rnks.size() != this->_shape.size()) {
      cytnx_error_msg(true, "%s",
                      "reshape a tensor with a specify shape that does not match with the shape of "
                      "the incident tensor.");
    }

    if (vec_unique(rnks).size() != rnks.size()) {
      cytnx_error_msg(true, "%s", "tensor permute with duplicated index.\n");
    }

    // std::vector<cytnx_uint64> new_fwdmap(this->_shape.size());
    // std::vector<cytnx_uint64> new_shape(this->_shape.size());
    // std::vector<cytnx_uint64> new_idxmap(this->_shape.size());

    // smallvec<cytnx_uint64> new_fwdmap(this->_shape.size());
    // smallvec<cytnx_uint64> new_shape(this->_shape.size());
    // smallvec<cytnx_uint64> new_idxmap(this->_shape.size());
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
      // new_idxmap[this->_mapper[rnks[i]]] = i;
      this->_invmapper[this->_mapper[rnks[i]]] = i;
      new_fwdmap[i] = this->_mapper[rnks[i]];
      new_shape[i] = this->_shape[rnks[i]];
    }

    // this->_invmapper = std::move(new_idxmap);
    for (cytnx_uint64 i = 0; i < this->_shape.size(); i++) {
      this->_shape[i] = new_shape[i];
      this->_mapper[i] = new_fwdmap[i];
    }

    // this->_shape = std::move(new_shape);
    // this->_mapper = std::move(new_fwdmap);

    /// checking if permute back to contiguous:
    bool iconti = true;
    for (cytnx_uint32 i = 0; i < rnks.size(); i++) {
      // if (this->_mapper[i] != this->_invmapper[i]) {
      //   iconti = false;
      //   break;
      // }
      if (this->_mapper[i] != i) {
        iconti = false;
        break;
      }
    }
    this->_contiguous = iconti;
  }

  // shadow new:
  //

  boost::intrusive_ptr<Tensor_impl> Tensor_impl::get(
    const std::vector<cytnx::Accessor> &accessors) {
    cytnx_error_msg(accessors.size() > this->_shape.size(), "%s",
                    "The input indexes rank is out of range! (>Tensor's rank).");

    std::vector<cytnx::Accessor> acc = accessors;
    for (int i = 0; i < this->_shape.size() - accessors.size(); i++) {
      acc.push_back(Accessor::all());
    }

    /*
    cout << "acc type bef" << endl;
    for(int i=0;i<acc.size();i++){
        cout << acc[i].type() << " ";
    }
    */
    acc = vec_map(acc, this->_invmapper);  // contiguous.
    /*
    cout << "acc type aft" << endl;
    for(int i=0;i<acc.size();i++){
        cout << acc[i].type() << " ";
    }
    */

    //[1] curr_shape:
    auto curr_shape = vec_map(this->_shape, this->_invmapper);
    // cout << "curr_shape" << endl;
    // cout << curr_shape << endl;

    //[2] from back to front, check until last all:
    cytnx_uint64 Nunit = 1;
    int tmpidx = 0;
    while (tmpidx < curr_shape.size()) {
      if (acc.back().type() == Accessor::All) {
        Nunit *= curr_shape[curr_shape.size() - 1 - tmpidx];
        tmpidx++;
        acc.pop_back();
      } else {
        break;
      }
    }
    // cout << "tmpidx" << tmpidx << endl;
    // cout << "Nunit" << Nunit << endl;
    // cout << acc.size() << endl;

    // acc-> locators

    std::vector<cytnx_uint64> get_shape(acc.size());
    std::vector<std::vector<cytnx_uint64>> locators(acc.size());
    for (cytnx_uint32 i = 0; i < acc.size(); i++) {
      cytnx_error_msg(acc[i].type() == Accessor::Qns,
                      "[ERROR] Tensor cannot accept accessor with qnum list.%s", "\n");
      acc[i].get_len_pos(curr_shape[i], get_shape[i], locators[i]);
    }
    // cout << "get_shape" << endl;
    // cout << get_shape << endl;

    // create Tensor:
    for (cytnx_uint64 i = 0; i < tmpidx; i++) {
      get_shape.push_back(curr_shape[acc.size() + i]);
    }
    boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());
    out->Init(get_shape, this->dtype(), this->device());
    // cout << get_shape << endl;

    if (locators.size() == 0) {
      locators.resize(1);
      locators[0].push_back(0);
    }

    // call storage
    this->storage()._impl->GetElem_byShape_v2(out->storage()._impl, curr_shape, locators, Nunit);

    // permute back:
    std::vector<cytnx_int64> new_mapper(this->_mapper.begin(), this->_mapper.end());
    std::vector<cytnx_int64> new_shape;
    std::vector<cytnx_int32> remove_id;
    for (unsigned int i = 0; i < out->_shape.size(); i++) {
      if (out->shape()[i] == 1 && (acc[i].type() == Accessor::Singl))
        remove_id.push_back(this->_mapper[this->_invmapper[i]]);
      else
        new_shape.push_back(out->shape()[i]);
    }

    // cout << "mapper" << endl;
    // cout << new_mapper << endl;
    // cout << "inv_mapper" << endl;
    // cout << this->_invmapper << endl;

    // cout << "remove_id" << endl;
    // cout << remove_id << endl;
    // cout << "out shape raw" << endl;
    // cout << out->shape() << endl;

    // cout << "perm" << endl;
    // cout << perm << endl;
    // cout << new_shape << endl;
    if (new_shape.size()) {  // exclude the case where only single element exists!

      out->reshape_(new_shape);  // remove size-1 axis

      std::vector<cytnx_uint64> perm;
      for (unsigned int i = 0; i < new_mapper.size(); i++) {
        perm.push_back(new_mapper[i]);
        for (unsigned int j = 0; j < remove_id.size(); j++) {
          if (new_mapper[i] > remove_id[j])
            perm.back() -= 1;
          else if (new_mapper[i] == remove_id[j]) {
            perm.pop_back();
            break;
          }
        }
      }
      out->permute_(perm);
    } else {
      out->reshape_({1});  // if it is only one element.
    }

    return out;
  }

  boost::intrusive_ptr<Tensor_impl> Tensor_impl::get_deprecated(
    const std::vector<cytnx::Accessor> &accessors) {
    cytnx_error_msg(accessors.size() > this->_shape.size(), "%s",
                    "The input indexes rank is out of range! (>Tensor's rank).");

    std::vector<cytnx::Accessor> acc = accessors;
    for (int i = 0; i < this->_shape.size() - accessors.size(); i++) {
      acc.push_back(Accessor::all());
    }

    vector<cytnx_uint64> get_shape(acc.size());

    // vector<cytnx_uint64> new_shape;
    std::vector<std::vector<cytnx_uint64>> locators(this->_shape.size());
    for (cytnx_uint32 i = 0; i < acc.size(); i++) {
      acc[i].get_len_pos(this->_shape[i], get_shape[i], locators[i]);
      // std::cout << this->_shape[i] << " " << get_shape[i] << "|";
      // for(int j=0;j<locators[i].size();j++) std::cout << locators[i][j] << " ";
      // std::cout << std::endl;
    }

    boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());
    out->Init(get_shape, this->dtype(), this->device());

    this->storage()._impl->GetElem_byShape(out->storage()._impl, this->shape(), this->_mapper,
                                           get_shape, locators);

    vector<cytnx_int64> new_shape;
    for (cytnx_uint32 i = 0; i < acc.size(); i++)
      if (get_shape[i] != 1) new_shape.push_back(get_shape[i]);

    if (new_shape.size() == 0)
      out->reshape_({1});
    else
      out->reshape_(new_shape);
    return out;
  }

  void Tensor_impl::set(const std::vector<cytnx::Accessor> &accessors,
                        const boost::intrusive_ptr<Tensor_impl> &rhs) {
    // cout << "calling set" << endl;
    cytnx_error_msg(accessors.size() > this->_shape.size(), "%s",
                    "The input indexes rank is out of range! (>Tensor's rank).");

    vector<cytnx::Accessor> acc = accessors;
    for (int i = 0; i < this->_shape.size() - accessors.size(); i++) {
      acc.push_back(Accessor::all());
    }

    // vector<cytnx_uint64> get_shape(acc.size());
    acc = vec_map(acc, this->_invmapper);  // contiguous.

    //[1] curr_shape:
    auto curr_shape = vec_map(this->_shape, this->_invmapper);

    //[2] from back to front, check until last all:
    cytnx_uint64 Nunit = 1;
    int tmpidx = 0;
    while (tmpidx < curr_shape.size()) {
      if (acc.back().type() == Accessor::All) {
        Nunit *= curr_shape[curr_shape.size() - 1 - tmpidx];
        tmpidx++;
        acc.pop_back();
      } else {
        break;
      }
    }

    std::vector<cytnx_uint64> get_shape(acc.size());
    std::vector<std::vector<cytnx_uint64>> locators(acc.size());
    for (cytnx_uint32 i = 0; i < acc.size(); i++) {
      cytnx_error_msg(acc[i].type() == Accessor::Qns,
                      "[ERROR] Tensor cannot accept accessor with qnum list.%s", "\n");
      acc[i].get_len_pos(curr_shape[i], get_shape[i], locators[i]);
    }

    /// checking if its scalar assign!
    if (rhs->storage().size() == 1) {
      this->storage()._impl->SetElem_byShape_v2(rhs->storage()._impl, curr_shape, locators, Nunit,
                                                true);
      // std::cout << "Scalar" << endl;

    } else {
      for (cytnx_uint64 i = 0; i < tmpidx; i++) {
        get_shape.push_back(curr_shape[acc.size() + i]);
      }

      // std::cout << get_shape << endl;

      // permute input to currect pos
      std::vector<cytnx_int64> new_mapper(this->_mapper.begin(), this->_mapper.end());
      std::vector<cytnx_uint64> new_shape;
      std::vector<cytnx_int32> remove_id;
      for (unsigned int i = 0; i < get_shape.size(); i++) {
        if (acc[i].type() == Accessor::Singl)
          remove_id.push_back(this->_mapper[this->_invmapper[i]]);
        else
          new_shape.push_back(get_shape[i]);
      }

      if (new_shape.size() == 0) new_shape.push_back(1);

      // use current size to infer rhs permutation.
      std::vector<cytnx_uint64> perm;
      for (unsigned int i = 0; i < new_mapper.size(); i++) {
        perm.push_back(new_mapper[i]);

        for (unsigned int j = 0; j < remove_id.size(); j++) {
          if (new_mapper[i] > remove_id[j])
            perm.back() -= 1;
          else if (new_mapper[i] == remove_id[j]) {
            perm.pop_back();
            break;
          }
        }
      }

      std::vector<cytnx_uint64> iperm(perm.size());
      for (unsigned int i = 0; i < iperm.size(); i++) iperm[perm[i]] = i;

      // std::cout << new_shape << endl;
      boost::intrusive_ptr<Tensor_impl> tmp;
      // std::cout << iperm << std::endl;
      tmp = rhs->permute(iperm)->contiguous();
      cytnx_error_msg(new_shape != tmp->shape(), "[ERROR][Tensor.set_elems]%s",
                      "inconsistent shape");
      this->storage()._impl->SetElem_byShape_v2(tmp->storage()._impl, curr_shape, locators, Nunit,
                                                false);
    }
  }

  template <class T>
  void Tensor_impl::set(const std::vector<cytnx::Accessor> &accessors, const T &rc) {
    cytnx_error_msg(accessors.size() > this->_shape.size(), "%s",
                    "The input indexes rank is out of range! (>Tensor's rank).");

    std::vector<cytnx::Accessor> acc = accessors;
    for (int i = 0; i < this->_shape.size() - accessors.size(); i++) {
      acc.push_back(Accessor::all());
    }

    acc = vec_map(acc, this->_invmapper);  // contiguous.

    //[1] curr_shape:
    auto curr_shape = vec_map(this->_shape, this->_invmapper);

    //[2] from back to front, check until last all:
    cytnx_uint64 Nunit = 1;
    int tmpidx = 0;
    while (tmpidx < curr_shape.size()) {
      if (acc.back().type() == Accessor::All) {
        Nunit *= curr_shape[curr_shape.size() - 1 - tmpidx];
        tmpidx++;
        acc.pop_back();
      } else {
        break;
      }
    }
    // cout << "tmpidx" << tmpidx << endl;
    // cout << "Nunit" << Nunit << endl;
    // cout << acc.size() << endl;

    // acc-> locators

    std::vector<cytnx_uint64> get_shape(acc.size());
    std::vector<std::vector<cytnx_uint64>> locators(acc.size());
    for (cytnx_uint32 i = 0; i < acc.size(); i++) {
      cytnx_error_msg(acc[i].type() == Accessor::Qns,
                      "[ERROR] Tensor cannot accept accessor with qnum list.%s", "\n");
      acc[i].get_len_pos(curr_shape[i], get_shape[i], locators[i]);
    }
    // cout << "get_shape" << endl;
    // cout << get_shape << endl;

    // call storage
    Scalar c = rc;

    Storage tmp(1, c.dtype(), this->device());
    tmp.set_item(0, rc);
    this->storage()._impl->SetElem_byShape_v2(tmp._impl, curr_shape, locators, Nunit, true);
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
  template void Tensor_impl::set<Scalar>(const std::vector<cytnx::Accessor> &, const Scalar &);

  void Tensor_impl::set(const std::vector<cytnx::Accessor> &accessors, const Scalar::Sproxy &rc) {
    this->set(accessors, Scalar(rc));
  }

}  // namespace cytnx
