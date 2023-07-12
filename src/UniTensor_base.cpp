#include "UniTensor.hpp"
#include "Tensor.hpp"
#include "linalg.hpp"
#include "Network.hpp"
#include <string>

namespace cytnx {
  //====================================================
  std::string UniTensorType_class::getname(const int &ut_type) {
    if (ut_type == this->Void) {
      return std::string("Void (un-initialize UniTensor");
    } else if (ut_type == this->Dense) {
      return std::string("Dense");
    } else if (ut_type == this->Sparse) {
      return std::string("Sparse ");
    } else if (ut_type == this->Block) {
      return std::string("Block ");
    } else {
      cytnx_error_msg(true, "%s\n", "[ERROR] invalid ut_type");
      return std::string("");
    }
    // extend more in here!!
  }
  UniTensorType_class UTenType;
  //===================================================
  /*
  void UniTensor_base::Init(const std::vector<Bond> &bonds,
                            const std::vector<cytnx_int64> &in_labels, const cytnx_int64 &rowrank,
                            const unsigned int &dtype, const int &device, const bool &is_diag,
                            const bool &no_alloc) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  */
  void UniTensor_base::Init(const std::vector<Bond> &bonds,
                            const std::vector<std::string> &in_labels, const cytnx_int64 &rowrank,
                            const unsigned int &dtype, const int &device, const bool &is_diag,
                            const bool &no_alloc, const std::string &name) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::Init_by_Tensor(const Tensor &in_tensor, const bool &is_diag,
                                      const cytnx_int64 &rowrank, const std::string &name) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  bool UniTensor_base::same_data(const boost::intrusive_ptr<UniTensor_base> &rhs) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return false;
  }
  std::vector<cytnx_uint64> UniTensor_base::shape() const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return std::vector<cytnx_uint64>();
  }
  void UniTensor_base::set_rowrank(const cytnx_uint64 &new_rowrank) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  boost::intrusive_ptr<UniTensor_base> UniTensor_base::astype(const unsigned int &dtype) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }
  bool UniTensor_base::is_blockform() const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return false;
  }
  bool UniTensor_base::is_contiguous() const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return false;
  }
  void UniTensor_base::to_(const int &device) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  boost::intrusive_ptr<UniTensor_base> UniTensor_base::to(const int &device) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }
  boost::intrusive_ptr<UniTensor_base> UniTensor_base::clone() const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }
  unsigned int UniTensor_base::dtype() const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return 0;
  }
  std::vector<Symmetry> UniTensor_base::syms() const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return std::vector<Symmetry>();
  }
  int UniTensor_base::device() const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return -1;
  }
  std::string UniTensor_base::dtype_str() const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return std::string("");
  }
  std::string UniTensor_base::device_str() const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return std::string("");
  }
  boost::intrusive_ptr<UniTensor_base> UniTensor_base::permute(
    const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }
  boost::intrusive_ptr<UniTensor_base> UniTensor_base::permute(
    const std::vector<std::string> &mapper, const cytnx_int64 &rowrank) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }

  void UniTensor_base::permute_(const std::vector<cytnx_int64> &mapper,
                                const cytnx_int64 &rowrank) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::permute_(const std::vector<std::string> &mapper,
                                const cytnx_int64 &rowrank) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

  boost::intrusive_ptr<UniTensor_base> UniTensor_base::contiguous_() {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  boost::intrusive_ptr<UniTensor_base> UniTensor_base::contiguous() {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    boost::intrusive_ptr<UniTensor_base> out(new UniTensor_base());
    return out;
  }
  void UniTensor_base::print_diagram(const bool &bond_info) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::print_blocks(const bool &full_info) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::print_block(const cytnx_int64 &idx, const bool &full_info) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  Tensor UniTensor_base::get_block(const cytnx_uint64 &idx) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return Tensor();
  }
  Tensor UniTensor_base::get_block(const std::vector<cytnx_int64> &qnum, const bool &force) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return Tensor();
  }
  // return a share view of block, this only work for non-symm tensor.
  const Tensor &UniTensor_base::get_block_(const cytnx_uint64 &idx) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    Tensor t;
    return t;
  }
  // return a share view of block, this only work for non-symm tensor.
  Tensor &UniTensor_base::get_block_(const cytnx_uint64 &idx) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    Tensor t;
    return t;
  }
  const Tensor &UniTensor_base::get_block_(const std::vector<cytnx_int64> &qnum,
                                           const bool &force) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    Tensor t;
    return t;
  }
  Tensor &UniTensor_base::get_block_(const std::vector<cytnx_int64> &qnum, const bool &force) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    Tensor t;
    return t;
  }
  std::vector<Tensor> UniTensor_base::get_blocks() const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return std::vector<Tensor>();
  }
  const std::vector<Tensor> &UniTensor_base::get_blocks_(const bool &silent = false) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return std::vector<Tensor>();
  }
  std::vector<Tensor> &UniTensor_base::get_blocks_(const bool &silent = false) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    std::vector<Tensor> t;
    return t;
  }

  void UniTensor_base::put_block(const Tensor &in, const cytnx_uint64 &idx) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::put_block_(Tensor &in, const cytnx_uint64 &idx) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::put_block(const Tensor &in, const std::vector<cytnx_int64> &qnum,
                                 const bool &force) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::put_block_(Tensor &in, const std::vector<cytnx_int64> &qnum,
                                  const bool &force) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::set(const std::vector<Accessor> &accesssors, const Tensor &rhs) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  boost::intrusive_ptr<UniTensor_base> UniTensor_base::get(
    const std::vector<Accessor> &accesssors) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }

  void UniTensor_base::reshape_(const std::vector<cytnx_int64> &new_shape,
                                const cytnx_uint64 &rowrank) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  boost::intrusive_ptr<UniTensor_base> UniTensor_base::reshape(
    const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &rowrank) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }

  void UniTensor_base::set_labels(const std::vector<std::string> &new_labels) {
    cytnx_error_msg(new_labels.size() != this->_labels.size(),
                    "[ERROR][set_labels][UniTensor] %s\n",
                    "the len(new_labels) does not match the rank of the UniTenosr");
    // check duplicate:
    std::vector<std::string> tmp = vec_unique<std::string>(new_labels);
    cytnx_error_msg(tmp.size() != new_labels.size(), "[ERROR][set_labels][UniTensor] %s\n",
                    "the input labels cannot contain duplicated element(s).");
    this->_labels = new_labels;
  }

  boost::intrusive_ptr<UniTensor_base> UniTensor_base::relabels(
    const std::vector<std::string> &new_labels) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }
  void UniTensor_base::relabels_(const std::vector<std::string> &new_labels) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

  boost::intrusive_ptr<UniTensor_base> UniTensor_base::relabel(const cytnx_int64 &inx,
                                                               const std::string &new_label) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }
  void UniTensor_base::relabel_(const cytnx_int64 &inx, const std::string &new_label) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

  boost::intrusive_ptr<UniTensor_base> UniTensor_base::relabel(const std::string &inx,
                                                               const std::string &new_label) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }
  void UniTensor_base::relabel_(const std::string &inx, const std::string &new_label) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

  boost::intrusive_ptr<UniTensor_base> UniTensor_base::to_dense() {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }
  void UniTensor_base::to_dense_() {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::combineBonds(const std::vector<cytnx_int64> &indicators,
                                    const bool &permute_back, const bool &by_labels) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::combineBonds(const std::vector<std::string> &indicators,
                                    const bool &permute_back) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::combineBonds(const std::vector<cytnx_int64> &indicators,
                                    const bool &permute_back) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  boost::intrusive_ptr<UniTensor_base> UniTensor_base::contract(
    const boost::intrusive_ptr<UniTensor_base> &rhs, const bool &mv_elem_self,
    const bool &mv_elem_rhs) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }
  std::vector<Bond> UniTensor_base::getTotalQnums(const bool &physical) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return std::vector<Bond>();
  }
  std::vector<std::vector<cytnx_int64>> UniTensor_base::get_blocks_qnums() const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return std::vector<std::vector<cytnx_int64>>();
  }
  boost::intrusive_ptr<UniTensor_base> UniTensor_base::Conj() {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }
  void UniTensor_base::Conj_() {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  boost::intrusive_ptr<UniTensor_base> UniTensor_base::normalize() {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }
  void UniTensor_base::normalize_() {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

  Tensor UniTensor_base::Norm() const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return Tensor();
  }

  boost::intrusive_ptr<UniTensor_base> UniTensor_base::Trace(const std::string &a,
                                                             const std::string &b) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }

  boost::intrusive_ptr<UniTensor_base> UniTensor_base::Trace(const cytnx_int64 &a,
                                                             const cytnx_int64 &b) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }
  void UniTensor_base::Trace_(const std::string &a, const std::string &b) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::Trace_(const cytnx_int64 &a, const cytnx_int64 &b) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

  void UniTensor_base::Transpose_() {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  boost::intrusive_ptr<UniTensor_base> UniTensor_base::Transpose() {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }

  void UniTensor_base::Dagger_() {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  boost::intrusive_ptr<UniTensor_base> UniTensor_base::Dagger() {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
    return nullptr;
  }

  void UniTensor_base::tag() {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  Scalar::Sproxy UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  cytnx_complex128 &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                  const cytnx_complex128 &aux) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  cytnx_complex64 &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                 const cytnx_complex64 &aux) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  cytnx_double &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                              const cytnx_double &aux) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  cytnx_float &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                             const cytnx_float &aux) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  cytnx_uint64 &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                              const cytnx_uint64 &aux) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  cytnx_int64 &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                             const cytnx_int64 &aux) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  cytnx_uint32 &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                              const cytnx_uint32 &aux) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  cytnx_int32 &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                             const cytnx_int32 &aux) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  cytnx_uint16 &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                              const cytnx_uint16 &aux) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  cytnx_int16 &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                             const cytnx_int16 &aux) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

  const Scalar::Sproxy UniTensor_base::at_for_sparse(
    const std::vector<cytnx_uint64> &locator) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  const cytnx_complex128 &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                        const cytnx_complex128 &aux) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  const cytnx_complex64 &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                       const cytnx_complex64 &aux) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  const cytnx_double &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                    const cytnx_double &aux) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  const cytnx_float &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                   const cytnx_float &aux) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  const cytnx_uint64 &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                    const cytnx_uint64 &aux) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  const cytnx_int64 &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                   const cytnx_int64 &aux) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  const cytnx_uint32 &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                    const cytnx_uint32 &aux) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  const cytnx_int32 &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                   const cytnx_int32 &aux) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  const cytnx_uint16 &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                    const cytnx_uint16 &aux) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  const cytnx_int16 &UniTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                   const cytnx_int16 &aux) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

  bool UniTensor_base::elem_exists(const std::vector<cytnx_uint64> &locator) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

  void UniTensor_base::_save_dispatch(std::fstream &f) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

  void UniTensor_base::_load_dispatch(std::fstream &f) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

  void UniTensor_base::truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::truncate_(const std::string &bond_idx, const cytnx_uint64 &dim) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

  //-----------------------------------------
  std::ostream &operator<<(std::ostream &os, const UniTensor &in) {
    in.print_blocks();
    return os;
  }

  UniTensor Contract(const UniTensor &inL, const UniTensor &inR, const bool &cacheL,
                     const bool &cacheR) {
    return inL.contract(inR, cacheL, cacheR);
  }

  void _resolve_CT(std::vector<UniTensor> &TNlist){};
  UniTensor Contracts(const std::vector<UniTensor> &TNs, const std::string &order,
                      const bool &optimal) {
    cytnx_error_msg(TNs.size() < 2, "[ERROR][Contracts] should have more than 2 TNs to contract.%s",
                    "\n");
    // UniTensor out = TNs[0].contract(TNs[1]);
    // if (TNs.size() > 2) {
    //   for (int i = 2; i < TNs.size(); i++) {
    //     out = out.contract(TNs[i]);
    //   }
    // }
    Network tmp;
    std::vector<std::vector<std::string>> lbls;
    std::vector<std::string> names;
    for (int i = 0; i < TNs.size(); i++) {
      names.push_back(TNs[i].name());
      lbls.push_back(TNs[i].labels());
    }
    tmp.construct(names, lbls, std::vector<std::string>(), -1, order, optimal);
    tmp.PutUniTensors(names, TNs);
    UniTensor out = tmp.Launch(optimal);
    return out;
  }

  // arithmetic
  void UniTensor_base::Add_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::Add_(const Scalar &rhs) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

  void UniTensor_base::Sub_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::Sub_(const Scalar &rhs) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::lSub_(const Scalar &lhs) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

  void UniTensor_base::Mul_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::Mul_(const Scalar &rhs) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

  void UniTensor_base::Div_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::Div_(const Scalar &rhs) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::lDiv_(const Scalar &lhs) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  void UniTensor_base::group_basis_() {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

  const std::vector<cytnx_uint64> &UniTensor_base::get_qindices(const cytnx_uint64 &bidx) const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  std::vector<cytnx_uint64> &UniTensor_base::get_qindices(const cytnx_uint64 &bidx) {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

  const vec2d<cytnx_uint64> &UniTensor_base::get_itoi() const {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }
  vec2d<cytnx_uint64> &UniTensor_base::get_itoi() {
    cytnx_error_msg(true, "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base%s",
                    "\n");
  }

}  // namespace cytnx
