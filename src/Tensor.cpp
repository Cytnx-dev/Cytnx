#include "Tensor.hpp"

#include <filesystem>
#include <typeinfo>

#include "linalg.hpp"
#include "utils/is.hpp"
#include "utils/checked_cast.hpp"
#include "Type.hpp"

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  namespace {
    constexpr unsigned int kLegacyTensorMagic = 888;
    constexpr unsigned int kVersionedTensorMagic = 889;
    constexpr unsigned int kCurrentTensorFileVersion = 1;
  }  // namespace

  //----------------------------------------------
  // Tproxy

  Tensor Tensor::Tproxy::operator+=(const Tensor::Tproxy &rc) {
    Tensor self;
    self._impl = _insimpl->get(_accs);
    // self += Tensor(rc);
    cytnx::linalg::iAdd(self, Tensor(rc));

    _insimpl->set(_accs, self._impl);
    self._impl = this->_insimpl;
    return self;
  }
  Tensor Tensor::Tproxy::operator-=(const Tensor::Tproxy &rc) {
    Tensor self;
    self._impl = _insimpl->get(_accs);
    // self += Tensor(rc);
    cytnx::linalg::iSub(self, Tensor(rc));

    _insimpl->set(_accs, self._impl);
    self._impl = this->_insimpl;
    return self;
  }
  Tensor Tensor::Tproxy::operator/=(const Tensor::Tproxy &rc) {
    Tensor self;
    self._impl = _insimpl->get(_accs);
    // self += Tensor(rc);
    cytnx::linalg::iDiv(self, Tensor(rc));

    _insimpl->set(_accs, self._impl);
    self._impl = this->_insimpl;
    return self;
  }
  Tensor Tensor::Tproxy::operator*=(const Tensor::Tproxy &rc) {
    Tensor self;
    self._impl = _insimpl->get(_accs);
    // self += Tensor(rc);
    cytnx::linalg::iMul(self, Tensor(rc));

    _insimpl->set(_accs, self._impl);
    self._impl = this->_insimpl;
    return self;
  }

  template <std::size_t... Is>
  Tensor::pointer_types void_ptr_to_variant_impl(void *p, unsigned int dtype,
                                                 std::index_sequence<Is...>) {
    // Lambda to select the correct type based on dtype
    Tensor::pointer_types result;
    (
      [&]() {
        if (dtype == Is) {
          using TargetType =
            std::variant_alternative_t<Is,
                                       typename Tensor::internal::exclude_first<Type_list>::type>;
          result = static_cast<TargetType *>(p);
        }
      }(),
      ...);  // Fold expression
    return result;
  }

  Tensor::pointer_types Tensor::ptr() const {
    cytnx_error_msg(this->dtype() == 0, "[ERROR] operation not allowed for empty (void) Tensor.%s",
                    "\n");
    // dtype()-1 here because we have removed void from the variant
    return void_ptr_to_variant_impl(this->_impl->_storage._impl->data(), this->dtype() - 1,
                                    std::make_index_sequence<std::variant_size_v<pointer_types>>{});
  }

  #ifdef UNI_GPU
  template <std::size_t... Is>
  Tensor::gpu_pointer_types gpu_void_ptr_to_variant_impl(void *p, unsigned int dtype,
                                                         std::index_sequence<Is...>) {
    // Lambda to select the correct type based on dtype
    Tensor::gpu_pointer_types result;
    (
      [&]() {
        if (dtype == Is) {
          using TargetType = std::variant_alternative_t<
            Is, typename Tensor::internal::exclude_first<Type_list_gpu>::type>;
          result = static_cast<TargetType *>(p);
        }
      }(),
      ...);  // Fold expression
    return result;
  }

  Tensor::gpu_pointer_types Tensor::gpu_ptr() const {
    cytnx_error_msg(this->dtype() == 0, "[ERROR] operation not allowed for empty (void) Tensor.%s",
                    "\n");
    // dtype()-1 here because we have removed void from the variant
    return gpu_void_ptr_to_variant_impl(
      this->_impl->_storage._impl->data(), this->dtype() - 1,
      std::make_index_sequence<std::variant_size_v<gpu_pointer_types>>{});
  }
  #endif  // UNI_GPU

  // ADD
  Tensor Tensor::Tproxy::operator+(
    const cytnx_complex128 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Add(rc);
  }
  Tensor Tensor::Tproxy::operator+(
    const cytnx_complex64 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Add(rc);
  }
  Tensor Tensor::Tproxy::operator+(
    const cytnx_double &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Add(rc);
  }
  Tensor Tensor::Tproxy::operator+(
    const cytnx_float &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Add(rc);
  }
  Tensor Tensor::Tproxy::operator+(
    const cytnx_uint64 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Add(rc);
  }
  Tensor Tensor::Tproxy::operator+(
    const cytnx_int64 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Add(rc);
  }
  Tensor Tensor::Tproxy::operator+(
    const cytnx_uint32 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Add(rc);
  }
  Tensor Tensor::Tproxy::operator+(
    const cytnx_int32 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Add(rc);
  }
  Tensor Tensor::Tproxy::operator+(
    const cytnx_uint16 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Add(rc);
  }
  Tensor Tensor::Tproxy::operator+(
    const cytnx_int16 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Add(rc);
  }
  Tensor Tensor::Tproxy::operator+(
    const cytnx_bool &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Add(rc);
  }

  Tensor Tensor::Tproxy::operator+(const Tproxy &rc) const {
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return cytnx::linalg::Add(out, Tensor(rc));
  }

  // SUB:
  Tensor Tensor::Tproxy::operator-(
    const cytnx_complex128 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Sub(rc);
  }
  Tensor Tensor::Tproxy::operator-(
    const cytnx_complex64 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Sub(rc);
  }
  Tensor Tensor::Tproxy::operator-(
    const cytnx_double &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Sub(rc);
  }
  Tensor Tensor::Tproxy::operator-(
    const cytnx_float &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Sub(rc);
  }
  Tensor Tensor::Tproxy::operator-(
    const cytnx_uint64 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Sub(rc);
  }
  Tensor Tensor::Tproxy::operator-(
    const cytnx_int64 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Sub(rc);
  }
  Tensor Tensor::Tproxy::operator-(
    const cytnx_uint32 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Sub(rc);
  }
  Tensor Tensor::Tproxy::operator-(
    const cytnx_int32 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Sub(rc);
  }
  Tensor Tensor::Tproxy::operator-(
    const cytnx_uint16 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Sub(rc);
  }
  Tensor Tensor::Tproxy::operator-(
    const cytnx_int16 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Sub(rc);
  }
  Tensor Tensor::Tproxy::operator-(
    const cytnx_bool &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Sub(rc);
  }
  Tensor Tensor::Tproxy::operator-(const Tproxy &rc) const {
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return cytnx::linalg::Sub(out, Tensor(rc));
  }
  Tensor Tensor::Tproxy::operator-() const {
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Mul(-1);
  }

  // MUL
  Tensor Tensor::Tproxy::operator*(
    const cytnx_complex128 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Mul(rc);
  }
  Tensor Tensor::Tproxy::operator*(
    const cytnx_complex64 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Mul(rc);
  }
  Tensor Tensor::Tproxy::operator*(
    const cytnx_double &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Mul(rc);
  }
  Tensor Tensor::Tproxy::operator*(
    const cytnx_float &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Mul(rc);
  }
  Tensor Tensor::Tproxy::operator*(
    const cytnx_uint64 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Mul(rc);
  }
  Tensor Tensor::Tproxy::operator*(
    const cytnx_int64 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Mul(rc);
  }
  Tensor Tensor::Tproxy::operator*(
    const cytnx_uint32 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Mul(rc);
  }
  Tensor Tensor::Tproxy::operator*(
    const cytnx_int32 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Mul(rc);
  }
  Tensor Tensor::Tproxy::operator*(
    const cytnx_uint16 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Mul(rc);
  }
  Tensor Tensor::Tproxy::operator*(
    const cytnx_int16 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Mul(rc);
  }
  Tensor Tensor::Tproxy::operator*(
    const cytnx_bool &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Mul(rc);
  }
  Tensor Tensor::Tproxy::operator*(const Tproxy &rc) const {
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return cytnx::linalg::Mul(out, Tensor(rc));
  }

  // DIV
  Tensor Tensor::Tproxy::operator/(
    const cytnx_complex128 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Div(rc);
  }
  Tensor Tensor::Tproxy::operator/(
    const cytnx_complex64 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Div(rc);
  }
  Tensor Tensor::Tproxy::operator/(
    const cytnx_double &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Div(rc);
  }
  Tensor Tensor::Tproxy::operator/(
    const cytnx_float &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Div(rc);
  }
  Tensor Tensor::Tproxy::operator/(
    const cytnx_uint64 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Div(rc);
  }
  Tensor Tensor::Tproxy::operator/(
    const cytnx_int64 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Div(rc);
  }
  Tensor Tensor::Tproxy::operator/(
    const cytnx_uint32 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Div(rc);
  }
  Tensor Tensor::Tproxy::operator/(
    const cytnx_int32 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Div(rc);
  }
  Tensor Tensor::Tproxy::operator/(
    const cytnx_uint16 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Div(rc);
  }
  Tensor Tensor::Tproxy::operator/(
    const cytnx_int16 &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Div(rc);
  }
  Tensor Tensor::Tproxy::operator/(
    const cytnx_bool &rc) const {  //{return this->_operatorADD(rc);};
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return out.Div(rc);
  }
  Tensor Tensor::Tproxy::operator/(const Tproxy &rc) const {
    Tensor out;
    out._impl = _insimpl->get(_accs);
    return cytnx::linalg::Div(out, Tensor(rc));
  }

  std::ostream &operator<<(std::ostream &os, const Tensor &in) {
    if (in.is_contiguous())
      in._impl->storage()._impl->PrintElem_byShape(os, in.shape());
    else
      in._impl->storage()._impl->PrintElem_byShape(os, in.shape(), in._impl->invmapper());
    return os;
  }
  std::ostream &operator<<(std::ostream &os, const Tensor::Tproxy &in) {
    os << Tensor(in) << std::endl;
    return os;
  }
  //===================================================================
  // wrapper

  void Tensor::Tofile(const std::string &fname) const {
    if (!this->is_contiguous()) {
      auto A = this->contiguous();
      A.storage().Tofile(fname);
    } else {
      this->_impl->_storage.Tofile(fname);
    }
  }
  void Tensor::Tofile(const char *fname) const {
    if (!this->is_contiguous()) {
      auto A = this->contiguous();
      A.storage().Tofile(fname);
    } else {
      this->_impl->_storage.Tofile(fname);
    }
  }
  void Tensor::Tofile(std::fstream &f) const {
    if (!this->is_contiguous()) {
      auto A = this->contiguous();
      A.storage().Tofile(f);
    } else {
      this->_impl->_storage.Tofile(f);
    }
  }
  void Tensor::Save(const std::string &fname) const {
    std::fstream f;
    if (std::filesystem::path(fname).has_extension()) {
      // filename extension is given
      f.open(fname, std::ios::out | std::ios::trunc | std::ios::binary);
    } else {
      // add filename extension
      cytnx_warning_msg(true,
                        "Missing file extension in fname '%s'. I am adding the extension '.cytn'. "
                        "This is deprecated, please provide the file extension in the future.\n",
                        fname.c_str());
      f.open((fname + ".cytn"), std::ios::out | std::ios::trunc | std::ios::binary);
    }
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->_Save(f);
    f.close();
  }
  void Tensor::Save(const char *fname) const { this->Save(std::string(fname)); }
  void Tensor::_Save(std::fstream &f) const {
    // header
    // check:
    cytnx_error_msg(!f.is_open(), "[ERROR] invalid fstream!.%s", "\n");

    unsigned int IDDs = kVersionedTensorMagic;
    f.write((char *)&IDDs, sizeof(unsigned int));
    unsigned int version = kCurrentTensorFileVersion;
    f.write((char *)&version, sizeof(unsigned int));

    cytnx_uint64 shp = this->shape().size();
    cytnx_uint64 Conti = this->is_contiguous();
    f.write((char *)&shp, sizeof(cytnx_uint64));

    f.write((char *)&Conti, sizeof(cytnx_uint64));
    if (shp != 0) {
      f.write((char *)this->_impl->_shape.data(), sizeof(cytnx_uint64) * shp);
      f.write((char *)this->_impl->_mapper.data(), sizeof(cytnx_uint64) * shp);
      f.write((char *)this->_impl->_invmapper.data(), sizeof(cytnx_uint64) * shp);
    }

    // pass to storage for save:
    this->_impl->_storage._Save(f);
  }

  Tensor Tensor::Fromfile(const std::string &fname, const unsigned int &dtype,
                          const cytnx_int64 &count) {
    return Tensor::from_storage(Storage::Fromfile(fname, dtype, count));
  }
  Tensor Tensor::Fromfile(const char *fname, const unsigned int &dtype, const cytnx_int64 &count) {
    return Tensor::from_storage(Storage::Fromfile(fname, dtype, count));
  }
  Tensor Tensor::Load(const std::string &fname) {
    Tensor out;
    std::fstream f;
    f.open(fname, std::ios::in | std::ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] Cannot open file '%s'.\n", fname.c_str());
    }
    out._Load(f);
    f.close();
    return out;
  }
  Tensor Tensor::Load(const char *fname) { return Tensor::Load(std::string(fname)); }
  void Tensor::_Load(std::fstream &f) {
    // header
    // check:
    cytnx_error_msg(!f.is_open(), "[ERROR] invalid fstream!.%s", "\n");

    unsigned int tmpIDDs;
    f.read((char *)&tmpIDDs, sizeof(unsigned int));
    if (tmpIDDs == kVersionedTensorMagic) {
      unsigned int version;
      f.read((char *)&version, sizeof(unsigned int));
      cytnx_error_msg(version != kCurrentTensorFileVersion,
                      "[ERROR][Tensor::_Load] Unsupported Tensor file format version '%u'.%s",
                      version, "\n");
    } else {
      cytnx_error_msg(tmpIDDs != kLegacyTensorMagic, "[ERROR] the object is not a cytnx tensor!%s",
                      "\n");
    }

    cytnx_uint64 shp;
    cytnx_uint64 Conti;
    f.read((char *)&shp, sizeof(cytnx_uint64));
    f.read((char *)&Conti, sizeof(cytnx_uint64));
    this->_impl->_contiguous = Conti;

    this->_impl->_shape.resize(shp);
    this->_impl->_mapper.resize(shp);
    this->_impl->_invmapper.resize(shp);
    if (shp != 0) {
      f.read((char *)this->_impl->_shape.data(), sizeof(cytnx_uint64) * shp);
      f.read((char *)this->_impl->_mapper.data(), sizeof(cytnx_uint64) * shp);
      f.read((char *)this->_impl->_invmapper.data(), sizeof(cytnx_uint64) * shp);
    }

    // pass to storage for save:
    this->_impl->_storage._Load(f);
  }

  Tensor Tensor::real() {
    Tensor out;
    out._impl = this->_impl->_clone_meta_only();
    out._impl->_storage = this->_impl->_storage.real();
    return out;
  };

  Tensor Tensor::imag() {
    Tensor out;
    out._impl = this->_impl->_clone_meta_only();
    out._impl->_storage = this->_impl->_storage.imag();
    return out;
  }

  namespace {
    // Wrap a scalar as a host-resident rank-0 Tensor so scalar in-place arithmetic reuses the
    // iAdd/iSub/iMul/iDiv kernels. The wrapper deliberately stays on the CPU even when the LHS is
    // on a GPU: scalar RHS kernels read it with a host-side dereference and pass it in by value, so
    // keeping it on the host avoids a per-call H2D copy. See #988.
    template <class T>
    Tensor scalar_as_rank0_tensor(const T &rc) {
      Tensor s({}, Type.cy_typeid(rc), Device.cpu);
      s.storage().at<T>(0) = rc;
      return s;
    }
    Tensor scalar_as_rank0_tensor(const Scalar &rc) {
      Tensor s({}, rc.dtype(), Device.cpu);
      s.item() = rc;  // Sproxy assignment
      return s;
    }
  }  // namespace

  ///@cond
  // +=
  template <>
  Tensor &Tensor::operator+=<Tensor>(const Tensor &rc) {
    cytnx::linalg::iAdd(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<Tensor::Tproxy>(const Tensor::Tproxy &rc) {
    cytnx::linalg::iAdd(*this, Tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_complex128>(const cytnx_complex128 &rc) {
    cytnx::linalg::iAdd(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_complex64>(const cytnx_complex64 &rc) {
    cytnx::linalg::iAdd(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_double>(const cytnx_double &rc) {
    cytnx::linalg::iAdd(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_float>(const cytnx_float &rc) {
    cytnx::linalg::iAdd(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_int64>(const cytnx_int64 &rc) {
    cytnx::linalg::iAdd(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_uint64>(const cytnx_uint64 &rc) {
    cytnx::linalg::iAdd(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_int32>(const cytnx_int32 &rc) {
    cytnx::linalg::iAdd(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_uint32>(const cytnx_uint32 &rc) {
    cytnx::linalg::iAdd(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_int16>(const cytnx_int16 &rc) {
    cytnx::linalg::iAdd(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_uint16>(const cytnx_uint16 &rc) {
    cytnx::linalg::iAdd(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_bool>(const cytnx_bool &rc) {
    cytnx::linalg::iAdd(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<Scalar>(const Scalar &rc) {
    cytnx::linalg::iAdd(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<Scalar::Sproxy>(const Scalar::Sproxy &rc) {
    return this->operator+=(Scalar(rc));
  }
  // -=
  template <>
  Tensor &Tensor::operator-=<Tensor>(const Tensor &rc) {
    cytnx::linalg::iSub(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<Tensor::Tproxy>(const Tensor::Tproxy &rc) {
    cytnx::linalg::iSub(*this, Tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_complex128>(const cytnx_complex128 &rc) {
    cytnx::linalg::iSub(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_complex64>(const cytnx_complex64 &rc) {
    cytnx::linalg::iSub(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_double>(const cytnx_double &rc) {
    cytnx::linalg::iSub(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_float>(const cytnx_float &rc) {
    cytnx::linalg::iSub(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_int64>(const cytnx_int64 &rc) {
    cytnx::linalg::iSub(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_uint64>(const cytnx_uint64 &rc) {
    cytnx::linalg::iSub(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_int32>(const cytnx_int32 &rc) {
    cytnx::linalg::iSub(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_uint32>(const cytnx_uint32 &rc) {
    cytnx::linalg::iSub(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_int16>(const cytnx_int16 &rc) {
    cytnx::linalg::iSub(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_uint16>(const cytnx_uint16 &rc) {
    cytnx::linalg::iSub(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_bool>(const cytnx_bool &rc) {
    cytnx::linalg::iSub(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<Scalar>(const Scalar &rc) {
    cytnx::linalg::iSub(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<Scalar::Sproxy>(const Scalar::Sproxy &rc) {
    return this->operator-=(Scalar(rc));
  }
  // *=
  template <>
  Tensor &Tensor::operator*=<Tensor>(const Tensor &rc) {
    cytnx::linalg::iMul(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<Tensor::Tproxy>(const Tensor::Tproxy &rc) {
    cytnx::linalg::iMul(*this, Tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_complex128>(const cytnx_complex128 &rc) {
    cytnx::linalg::iMul(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_complex64>(const cytnx_complex64 &rc) {
    cytnx::linalg::iMul(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_double>(const cytnx_double &rc) {
    cytnx::linalg::iMul(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_float>(const cytnx_float &rc) {
    cytnx::linalg::iMul(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_int64>(const cytnx_int64 &rc) {
    cytnx::linalg::iMul(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_uint64>(const cytnx_uint64 &rc) {
    cytnx::linalg::iMul(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_int32>(const cytnx_int32 &rc) {
    cytnx::linalg::iMul(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_uint32>(const cytnx_uint32 &rc) {
    cytnx::linalg::iMul(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_int16>(const cytnx_int16 &rc) {
    cytnx::linalg::iMul(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_uint16>(const cytnx_uint16 &rc) {
    cytnx::linalg::iMul(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_bool>(const cytnx_bool &rc) {
    cytnx::linalg::iMul(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<Scalar>(const Scalar &rc) {
    cytnx::linalg::iMul(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<Scalar::Sproxy>(const Scalar::Sproxy &rc) {
    return this->operator*=(Scalar(rc));
  }

  // /=
  template <>
  Tensor &Tensor::operator/=<Tensor>(const Tensor &rc) {
    cytnx::linalg::iDiv(*this, rc);
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<Tensor::Tproxy>(const Tensor::Tproxy &rc) {
    cytnx::linalg::iDiv(*this, Tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_complex128>(const cytnx_complex128 &rc) {
    cytnx::linalg::iDiv(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_complex64>(const cytnx_complex64 &rc) {
    cytnx::linalg::iDiv(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_double>(const cytnx_double &rc) {
    cytnx::linalg::iDiv(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_float>(const cytnx_float &rc) {
    cytnx::linalg::iDiv(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_int64>(const cytnx_int64 &rc) {
    cytnx::linalg::iDiv(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_uint64>(const cytnx_uint64 &rc) {
    cytnx::linalg::iDiv(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_int32>(const cytnx_int32 &rc) {
    cytnx::linalg::iDiv(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_uint32>(const cytnx_uint32 &rc) {
    cytnx::linalg::iDiv(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_int16>(const cytnx_int16 &rc) {
    cytnx::linalg::iDiv(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_uint16>(const cytnx_uint16 &rc) {
    cytnx::linalg::iDiv(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_bool>(const cytnx_bool &rc) {
    cytnx::linalg::iDiv(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<Scalar>(const Scalar &rc) {
    cytnx::linalg::iDiv(*this, scalar_as_rank0_tensor(rc));
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<Scalar::Sproxy>(const Scalar::Sproxy &rc) {
    return this->operator/=(Scalar(rc));
  }
  ///@endcond

  // std::vector<Tensor> Tensor::Svd(const bool &is_U, const bool &is_vT) const {
  //   return linalg::Svd(*this, is_U, is_vT);
  // }
  std::vector<Tensor> Tensor::Svd(const bool &is_UvT) const { return linalg::Svd(*this, is_UvT); }
  std::vector<Tensor> Tensor::Eigh(const bool &is_V, const bool &row_v) const {
    return linalg::Eigh(*this, is_V, row_v);
  }

  Tensor &Tensor::InvM_() {
    linalg::InvM_(*this);
    return *this;
  }
  Tensor Tensor::InvM() const { return linalg::InvM(*this); }
  Tensor &Tensor::Inv_(const double &clip) {
    linalg::Inv_(*this, clip);
    return *this;
  }
  Tensor Tensor::Inv(const double &clip) const { return linalg::Inv(*this, clip); }

  Tensor &Tensor::Conj_() {
    linalg::Conj_(*this);
    return *this;
  }
  Tensor Tensor::Conj() const { return linalg::Conj(*this); }

  Tensor &Tensor::Exp_() {
    linalg::Exp_(*this);
    return *this;
  }
  Tensor Tensor::Exp() const { return linalg::Exp(*this); }
  Tensor Tensor::Norm() const { return linalg::Norm(*this); }

  Scalar Tensor::norm() const { return linalg::norm(*this); }

  Tensor Tensor::Pow(const cytnx_double &p) const { return linalg::Pow(*this, p); }

  Tensor &Tensor::Pow_(const cytnx_double &p) {
    linalg::Pow_(*this, p);
    return *this;
  }

  Tensor &Tensor::Abs_() {
    linalg::Abs_(*this);
    return *this;
  }
  Tensor Tensor::Abs() const { return linalg::Abs(*this); }
  Tensor Tensor::Max() const { return linalg::Max(*this); }
  Tensor Tensor::Min() const { return linalg::Min(*this); }

  Tensor Tensor::Trace(const cytnx_uint64 &a, const cytnx_uint64 &b) const {
    Tensor out = linalg::Trace(*this, a, b);
    return out;
  }

  bool Tensor::same_data(const Tensor &rhs) const {
    return is(this->_impl->storage(), rhs.storage());
  }

  std::vector<cytnx_int64> Tensor::strides() const {
    // The storage is laid out contiguously in memory order; _invmapper[i] gives
    // the logical axis sitting at memory position i (innermost last). The stride
    // of a logical axis is the product of the memory-order extents inside it.
    const std::vector<cytnx_uint64> &shape = this->_impl->shape();
    const std::vector<cytnx_uint64> &invmapper = this->_impl->invmapper();
    const cytnx_uint64 rank = shape.size();
    std::vector<cytnx_int64> out(rank);
    cytnx_uint64 step = 1;
    for (cytnx_int64 i = static_cast<cytnx_int64>(rank) - 1; i >= 0; i--) {
      out[invmapper[i]] = cytnx::internal::CheckedCastToInt64(step, "stride");
      step *= shape[invmapper[i]];
    }
    return out;
  }

  //===========================
  // Tensor am Tproxy
  Tensor operator+(const Tensor &lhs, const Tensor::Tproxy &rhs) {
    return cytnx::linalg::Add(lhs, Tensor(rhs));
  }
  Tensor operator-(const Tensor &lhs, const Tensor::Tproxy &rhs) {
    return cytnx::linalg::Sub(lhs, Tensor(rhs));
  }
  Tensor operator*(const Tensor &lhs, const Tensor::Tproxy &rhs) {
    return cytnx::linalg::Mul(lhs, Tensor(rhs));
  }
  Tensor operator/(const Tensor &lhs, const Tensor::Tproxy &rhs) {
    return cytnx::linalg::Div(lhs, Tensor(rhs));
  }

  //===========================
  // Tensor am Sproxy
  Tensor operator+(const Tensor &lhs, const Scalar::Sproxy &rhs) {
    return cytnx::linalg::Add(lhs, Scalar(rhs));
  }
  Tensor operator-(const Tensor &lhs, const Scalar::Sproxy &rhs) {
    return cytnx::linalg::Sub(lhs, Scalar(rhs));
  }
  Tensor operator*(const Tensor &lhs, const Scalar::Sproxy &rhs) {
    return cytnx::linalg::Mul(lhs, Scalar(rhs));
  }
  Tensor operator/(const Tensor &lhs, const Scalar::Sproxy &rhs) {
    return cytnx::linalg::Div(lhs, Scalar(rhs));
  }

}  // namespace cytnx
#endif
