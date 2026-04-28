#include "Tensor.hpp"

#include <filesystem>
#include <typeinfo>

#include "H5Cpp.h"
#include "Type.hpp"
#include "linalg.hpp"
#include "utils/is.hpp"

using namespace std;

#ifdef BACKEND_TORCH
#else

namespace cytnx {

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
      this->storage().Tofile(fname);
    }
  }
  void Tensor::Tofile(const char *fname) const {
    if (!this->is_contiguous()) {
      auto A = this->contiguous();
      A.storage().Tofile(fname);
    } else {
      this->storage().Tofile(fname);
    }
  }
  void Tensor::Tofile(fstream &f) const {
    if (!this->is_contiguous()) {
      auto A = this->contiguous();
      A.storage().Tofile(f);
    } else {
      this->storage().Tofile(f);
    }
  }
  void Tensor::Save(const std::string &fname) const {
    fstream f;  // only for binary saving, not used for hdf5
    if (std::filesystem::path(fname).has_extension()) {
      // filename extension is given
      auto ext = std::filesystem::path(fname).extension().string();
      if (ext == ".h5" || ext == ".hdf5" || ext == ".H5" || ext == ".HDF5" || ext == ".hdf" ||
          ext == ".HDF") {
        // save as hdf5
        H5::H5File h5file;
        try {
          h5file = H5::H5File(fname, H5F_ACC_TRUNC);
        } catch (H5::FileIException &error) {
          error.printErrorStack();
          cytnx_error_msg(true, "[ERROR] Cannot create HDF5 file '%s'.\n", fname.c_str());
        }
        this->to_hdf5(h5file);
        h5file.close();
        return;
      } else {  // create binary file
        f.open(fname, ios::out | ios::trunc | ios::binary);
      }
    } else {  // create binary file with standard extension
      cytnx_warning_msg(true,
                        "Missing file extension in fname '%s'. I am adding the extension '.cytn'. "
                        "This is deprecated, please provide the file extension in the future.\n",
                        fname.c_str());
      f.open((fname + ".cytn"), ios::out | ios::trunc | ios::binary);
    }
    // write binary
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->to_binary(f);
    f.close();
  }
  void Tensor::Save(const char *fname) const { this->Save(string(fname)); }

  void Tensor::to_hdf5(H5::Group &location, const std::string &name) const {
    Tensor ten = this->contiguous();
    std::vector<hsize_t> dims(this->shape().begin(), this->shape().end());

    H5::DataSpace dataspace(dims.size(), dims.data());
    H5::DataType datatype = Type.to_hdf5_type(this->dtype());
    H5::DataSet dataset = location.createDataSet(name, datatype, dataspace);
    ten.storage().data_to_hdf5(dataset, datatype);

    if (this->device() != Device.cpu) {
      H5::Attribute attr =
        dataset.createAttribute("device", H5::PredType::NATIVE_INT, H5::DataSpace(H5S_SCALAR));
      int device = this->device();
      attr.write(H5::PredType::NATIVE_INT, &device);
    }
  }
  void Tensor::to_binary(std::ostream &f) const {
    unsigned int IDDs = 888;
    f.write((char *)&IDDs, sizeof(unsigned int));
    cytnx_uint64 shp = this->shape().size();
    cytnx_uint64 Conti = this->is_contiguous();
    f.write((char *)&shp, sizeof(cytnx_uint64));

    f.write((char *)&Conti, sizeof(cytnx_uint64));
    f.write((char *)&this->_impl->_shape[0], sizeof(cytnx_uint64) * shp);
    f.write((char *)&this->_impl->_mapper[0], sizeof(cytnx_uint64) * shp);
    f.write((char *)&this->_impl->_invmapper[0], sizeof(cytnx_uint64) * shp);

    // pass to storage for save:
    this->storage().to_binary(f);
  }

  Tensor Tensor::Fromfile(const std::string &fname, const unsigned int &dtype,
                          const cytnx_int64 &count, const bool restore_device) {
    return Tensor::from_storage(Storage::Fromfile(fname, dtype, count, restore_device));
  }
  Tensor Tensor::Fromfile(const char *fname, const unsigned int &dtype, const cytnx_int64 &count,
                          const bool restore_device) {
    return Tensor::from_storage(Storage::Fromfile(fname, dtype, count, restore_device));
  }
  Tensor Tensor::Load(const std::string &fname, const bool restore_device) {
    Tensor out;
    out.Load_(fname, restore_device);
    return out;
  }
  Tensor Tensor::Load(const char *fname, const bool restore_device) {
    return Tensor::Load(string(fname), restore_device);
  }

  void Tensor::Load_(const std::string &fname, const bool restore_device) {
    auto ext = std::filesystem::path(fname).extension().string();
    if (ext == ".h5" || ext == ".hdf5" || ext == ".H5" || ext == ".HDF5" || ext == ".hdf" ||
        ext == ".HDF") {
      // load hdf5
      H5::H5File h5file;
      try {
        h5file = H5::H5File(fname, H5F_ACC_RDONLY);
      } catch (H5::FileIException &error) {
        error.printErrorStack();
        cytnx_error_msg(true, "[ERROR] Cannot open HDF5 file '%s'.\n", fname.c_str());
      }
      this->from_hdf5(h5file, "Tensor", restore_device);
      h5file.close();
    } else {  // load binary
      fstream f;
      f.open(fname, ios::in | ios::binary);
      if (!f.is_open()) {
        cytnx_error_msg(true, "[ERROR] Cannot open file '%s'.\n", fname.c_str());
      }
      this->from_binary(f, restore_device);
      f.close();
    }
  }
  void Tensor::Load_(const char *fname, const bool restore_device) {
    this->Load_(string(fname), restore_device);
  }

  void Tensor::from_hdf5(H5::Group &location, const std::string &name, const bool restore_device) {
    H5::DataSet dataset = location.openDataSet(name);
    H5::DataType datatype = dataset.getDataType();
    unsigned int dtype = Type.from_hdf5_type(datatype);
    H5::DataSpace dataspace = dataset.getSpace();
    int rank = dataspace.getSimpleExtentNdims();
    hsize_t dims[rank];
    dataspace.getSimpleExtentDims(dims);
    auto Nelem = dataspace.getSimpleExtentNpoints();

    this->_impl->_shape = std::vector<cytnx_uint64>(dims, dims + rank);
    this->_impl->_mapper = vec_range(this->_impl->_shape.size());
    this->_impl->_invmapper = this->_impl->_mapper;
    this->_impl->_contiguous = true;  // HDF5 data is always stored in contiguous layout

    int device = Device.cpu;
    if (restore_device && dataset.attrExists("device")) {
      H5::Attribute attr = dataset.openAttribute("device");
      attr.read(H5::PredType::NATIVE_INT, &device);
    }

    this->_impl->_storage.data_from_hdf5(dataset, Nelem, dtype, datatype, device);
  }
  void Tensor::from_binary(std::istream &f, const bool restore_device) {
    unsigned int tmpIDDs;
    f.read((char *)&tmpIDDs, sizeof(unsigned int));
    cytnx_error_msg(tmpIDDs != 888, "[ERROR] the object is not a cytnx tensor!%s", "\n");

    cytnx_uint64 shp;
    cytnx_uint64 Conti;
    f.read((char *)&shp, sizeof(cytnx_uint64));
    f.read((char *)&Conti, sizeof(cytnx_uint64));
    this->_impl->_contiguous = Conti;

    this->_impl->_shape.resize(shp);
    this->_impl->_mapper.resize(shp);
    this->_impl->_invmapper.resize(shp);
    f.read((char *)&this->_impl->_shape[0], sizeof(cytnx_uint64) * shp);
    f.read((char *)&this->_impl->_mapper[0], sizeof(cytnx_uint64) * shp);
    f.read((char *)&this->_impl->_invmapper[0], sizeof(cytnx_uint64) * shp);

    // pass to storage for save:
    this->_impl->_storage.from_binary(f, restore_device);
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
    this->_impl->storage() = cytnx::linalg::Add(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_complex64>(const cytnx_complex64 &rc) {
    this->_impl->storage() = cytnx::linalg::Add(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_double>(const cytnx_double &rc) {
    this->_impl->storage() = cytnx::linalg::Add(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_float>(const cytnx_float &rc) {
    this->_impl->storage() = cytnx::linalg::Add(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_int64>(const cytnx_int64 &rc) {
    this->_impl->storage() = cytnx::linalg::Add(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_uint64>(const cytnx_uint64 &rc) {
    this->_impl->storage() = cytnx::linalg::Add(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_int32>(const cytnx_int32 &rc) {
    this->_impl->storage() = cytnx::linalg::Add(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_uint32>(const cytnx_uint32 &rc) {
    this->_impl->storage() = cytnx::linalg::Add(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_int16>(const cytnx_int16 &rc) {
    this->_impl->storage() = cytnx::linalg::Add(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_uint16>(const cytnx_uint16 &rc) {
    this->_impl->storage() = cytnx::linalg::Add(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<cytnx_bool>(const cytnx_bool &rc) {
    this->_impl->storage() = cytnx::linalg::Add(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator+=<Scalar>(const Scalar &rc) {
    this->_impl->storage() = cytnx::linalg::Add(*this, rc)._impl->storage();
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
    this->_impl->storage() = cytnx::linalg::Sub(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_complex64>(const cytnx_complex64 &rc) {
    this->_impl->storage() = cytnx::linalg::Sub(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_double>(const cytnx_double &rc) {
    this->_impl->storage() = cytnx::linalg::Sub(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_float>(const cytnx_float &rc) {
    this->_impl->storage() = cytnx::linalg::Sub(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_int64>(const cytnx_int64 &rc) {
    this->_impl->storage() = cytnx::linalg::Sub(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_uint64>(const cytnx_uint64 &rc) {
    this->_impl->storage() = cytnx::linalg::Sub(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_int32>(const cytnx_int32 &rc) {
    this->_impl->storage() = cytnx::linalg::Sub(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_uint32>(const cytnx_uint32 &rc) {
    this->_impl->storage() = cytnx::linalg::Sub(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_int16>(const cytnx_int16 &rc) {
    this->_impl->storage() = cytnx::linalg::Sub(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_uint16>(const cytnx_uint16 &rc) {
    this->_impl->storage() = cytnx::linalg::Sub(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<cytnx_bool>(const cytnx_bool &rc) {
    this->_impl->storage() = cytnx::linalg::Sub(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator-=<Scalar>(const Scalar &rc) {
    this->_impl->storage() = cytnx::linalg::Sub(*this, rc)._impl->storage();
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
    this->_impl->storage() = cytnx::linalg::Mul(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_complex64>(const cytnx_complex64 &rc) {
    this->_impl->storage() = cytnx::linalg::Mul(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_double>(const cytnx_double &rc) {
    this->_impl->storage() = cytnx::linalg::Mul(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_float>(const cytnx_float &rc) {
    this->_impl->storage() = cytnx::linalg::Mul(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_int64>(const cytnx_int64 &rc) {
    this->_impl->storage() = cytnx::linalg::Mul(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_uint64>(const cytnx_uint64 &rc) {
    this->_impl->storage() = cytnx::linalg::Mul(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_int32>(const cytnx_int32 &rc) {
    this->_impl->storage() = cytnx::linalg::Mul(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_uint32>(const cytnx_uint32 &rc) {
    this->_impl->storage() = cytnx::linalg::Mul(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_int16>(const cytnx_int16 &rc) {
    this->_impl->storage() = cytnx::linalg::Mul(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_uint16>(const cytnx_uint16 &rc) {
    this->_impl->storage() = cytnx::linalg::Mul(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<cytnx_bool>(const cytnx_bool &rc) {
    this->_impl->storage() = cytnx::linalg::Mul(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator*=<Scalar>(const Scalar &rc) {
    this->_impl->storage() = cytnx::linalg::Mul(*this, rc)._impl->storage();
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
    this->_impl->storage() = cytnx::linalg::Div(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_complex64>(const cytnx_complex64 &rc) {
    this->_impl->storage() = cytnx::linalg::Div(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_double>(const cytnx_double &rc) {
    this->_impl->storage() = cytnx::linalg::Div(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_float>(const cytnx_float &rc) {
    this->_impl->storage() = cytnx::linalg::Div(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_int64>(const cytnx_int64 &rc) {
    this->_impl->storage() = cytnx::linalg::Div(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_uint64>(const cytnx_uint64 &rc) {
    this->_impl->storage() = cytnx::linalg::Div(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_int32>(const cytnx_int32 &rc) {
    // std::cout << "entry /= int32" << std::endl;
    this->_impl->storage() = cytnx::linalg::Div(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_uint32>(const cytnx_uint32 &rc) {
    this->_impl->storage() = cytnx::linalg::Div(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_int16>(const cytnx_int16 &rc) {
    this->_impl->storage() = cytnx::linalg::Div(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_uint16>(const cytnx_uint16 &rc) {
    this->_impl->storage() = cytnx::linalg::Div(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<cytnx_bool>(const cytnx_bool &rc) {
    this->_impl->storage() = cytnx::linalg::Div(*this, rc)._impl->storage();
    return *this;
  }
  template <>
  Tensor &Tensor::operator/=<Scalar>(const Scalar &rc) {
    this->_impl->storage() = cytnx::linalg::Div(*this, rc)._impl->storage();
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
