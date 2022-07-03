#include "Storage.hpp"
#include "utils/utils_internal_interface.hpp"

using namespace std;

namespace cytnx {

  // Storage Init interface.
  //=============================
  boost::intrusive_ptr<Storage_base> SIInit_cd() {
    boost::intrusive_ptr<Storage_base> out(new ComplexDoubleStorage());
    return out;
  }
  boost::intrusive_ptr<Storage_base> SIInit_cf() {
    boost::intrusive_ptr<Storage_base> out(new ComplexFloatStorage());
    return out;
  }
  boost::intrusive_ptr<Storage_base> SIInit_d() {
    boost::intrusive_ptr<Storage_base> out(new DoubleStorage());
    return out;
  }
  boost::intrusive_ptr<Storage_base> SIInit_f() {
    boost::intrusive_ptr<Storage_base> out(new FloatStorage());
    return out;
  }
  boost::intrusive_ptr<Storage_base> SIInit_u64() {
    boost::intrusive_ptr<Storage_base> out(new Uint64Storage());
    return out;
  }
  boost::intrusive_ptr<Storage_base> SIInit_i64() {
    boost::intrusive_ptr<Storage_base> out(new Int64Storage());
    return out;
  }
  boost::intrusive_ptr<Storage_base> SIInit_u32() {
    boost::intrusive_ptr<Storage_base> out(new Uint32Storage());
    return out;
  }
  boost::intrusive_ptr<Storage_base> SIInit_i32() {
    boost::intrusive_ptr<Storage_base> out(new Int32Storage());
    return out;
  }
  boost::intrusive_ptr<Storage_base> SIInit_u16() {
    boost::intrusive_ptr<Storage_base> out(new Uint16Storage());
    return out;
  }
  boost::intrusive_ptr<Storage_base> SIInit_i16() {
    boost::intrusive_ptr<Storage_base> out(new Int16Storage());
    return out;
  }
  boost::intrusive_ptr<Storage_base> SIInit_b() {
    boost::intrusive_ptr<Storage_base> out(new BoolStorage());
    return out;
  }

  Storage_init_interface::Storage_init_interface() {
    USIInit.resize(N_Type);
    USIInit[this->Double] = SIInit_d;
    USIInit[this->Float] = SIInit_f;
    USIInit[this->ComplexDouble] = SIInit_cd;
    USIInit[this->ComplexFloat] = SIInit_cf;
    USIInit[this->Uint64] = SIInit_u64;
    USIInit[this->Int64] = SIInit_i64;
    USIInit[this->Uint32] = SIInit_u32;
    USIInit[this->Int32] = SIInit_i32;
    USIInit[this->Uint16] = SIInit_u16;
    USIInit[this->Int16] = SIInit_i16;
    USIInit[this->Bool] = SIInit_b;
  }

  //==========================
  void Storage_base::Init(const unsigned long long &len_in, const int &device) {
    // cout << "Base.init" << endl;
  }

  Storage_base::Storage_base(const unsigned long long &len_in, const int &device) {
    this->Init(len_in, device);
  }

  Storage_base &Storage_base::operator=(Storage_base &Rhs) {
    cout << "dev" << endl;
    return *this;
  }

  Storage_base::Storage_base(Storage_base &Rhs) { cout << "dev" << endl; }

  boost::intrusive_ptr<Storage_base> Storage_base::astype(const unsigned int &dtype) {
    boost::intrusive_ptr<Storage_base> out(new Storage_base());
    if (dtype == this->dtype) return boost::intrusive_ptr<Storage_base>(this);

    if (this->device == Device.cpu) {
      if (utils_internal::uii.ElemCast[this->dtype][dtype] == NULL) {
        cytnx_error_msg(1, "[ERROR] not support type with dtype=%d", dtype);
      } else {
        utils_internal::uii.ElemCast[this->dtype][dtype](this, out, this->len, 1);
      }
    } else {
#ifdef UNI_GPU
      if (utils_internal::uii.cuElemCast[this->dtype][dtype] == NULL) {
        cytnx_error_msg(1, "[ERROR] not support type with dtype=%d", dtype);
      } else {
        // std::cout << this->device << std::endl;
        utils_internal::uii.cuElemCast[this->dtype][dtype](this, out, this->len, this->device);
      }
#else
      cytnx_error_msg(
        1, "%s",
        "[ERROR][Internal Error] enter GPU section without CUDA support @ Storage.astype()");
#endif
    }
    return out;
  }

  boost::intrusive_ptr<Storage_base> Storage_base::_create_new_sametype() {
    cytnx_error_msg(1, "%s", "[ERROR] call _create_new_sametype in base");
  }

  boost::intrusive_ptr<Storage_base> Storage_base::clone() {
    boost::intrusive_ptr<Storage_base> out(new Storage_base());
    return out;
  }

  string Storage_base::dtype_str() const { return Type.getname(this->dtype); }
  string Storage_base::device_str() const { return Device.getname(this->device); }
  void Storage_base::_Init_byptr(void *rawptr, const unsigned long long &len_in,
                                 const int &device) {
    cytnx_error_msg(1, "%s", "[ERROR] call _Init_byptr in base");
  }

  Storage_base::~Storage_base() {
    // cout << "delet" << endl;
    if (Mem != NULL) {
      if (this->device == Device.cpu) {
        free(Mem);
      } else {
#ifdef UNI_GPU
        cudaFree(Mem);
#else
        cytnx_error_msg(1, "%s", "[ERROR] trying to free an GPU memory without CUDA install");
#endif
      }
    }
  }

  void Storage_base::Move_memory_(const std::vector<cytnx_uint64> &old_shape,
                                  const std::vector<cytnx_uint64> &mapper,
                                  const std::vector<cytnx_uint64> &invmapper) {
    cytnx_error_msg(1, "%s", "[ERROR] call Move_memory_ directly on Void Storage.");
  }

  boost::intrusive_ptr<Storage_base> Storage_base::Move_memory(
    const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper,
    const std::vector<cytnx_uint64> &invmapper) {
    cytnx_error_msg(1, "%s", "[ERROR] call Move_memory_ directly on Void Storage.");
  }

  void Storage_base::to_(const int &device) {
    cytnx_error_msg(1, "%s", "[ERROR] call to_ directly on Void Storage.");
  }

  boost::intrusive_ptr<Storage_base> Storage_base::to(const int &device) {
    cytnx_error_msg(1, "%s", "[ERROR] call to directly on Void Storage.");
  }

  void Storage_base::PrintElem_byShape(std::ostream &os, const std::vector<cytnx_uint64> &shape,
                                       const std::vector<cytnx_uint64> &mapper) {
    cytnx_error_msg(1, "%s", "[ERROR] call PrintElem_byShape directly on Void Storage.");
  }

  void Storage_base::print_info() {
    cout << "dtype : " << this->dtype_str() << endl;
    cout << "device: " << Device.getname(this->device) << endl;
    cout << "size  : " << this->len << endl;
  }
  void Storage_base::print_elems() {
    cytnx_error_msg(1, "%s", "[ERROR] call print_elems directly on Void Storage.");
  }
  void Storage_base::print() {
    this->print_info();
    this->print_elems();
  }
  void Storage_base::GetElem_byShape(boost::intrusive_ptr<Storage_base> &out,
                                     const std::vector<cytnx_uint64> &shape,
                                     const std::vector<cytnx_uint64> &mapper,
                                     const std::vector<cytnx_uint64> &len,
                                     const std::vector<std::vector<cytnx_uint64>> &locators) {
#ifdef UNI_DEBUG
    cytnx_error_msg(shape.size() != len.size(), "%s",
                    "[ERROR][DEBUG] internal Storage, shape.size() != len.size()");
    cytnx_error_msg(out->dtype != this->dtype, "%s", "[ERROR][DEBUG] %s",
                    "internal, the output dtype does not match current storage dtype.\n");
#endif

    // std::cout <<"=====" << len.size() << " " << locators.size() << std::endl;
    // create new instance:
    cytnx_uint64 TotalElem = 1;
    for (cytnx_uint32 i = 0; i < len.size(); i++) TotalElem *= len[i];

    cytnx_error_msg(out->size() != TotalElem, "%s",
                    "[ERROR] internal, the out Storage size does not match the no. of elems "
                    "calculated from Accessors.%s",
                    "\n");

    std::vector<cytnx_uint64> c_offj(shape.size());
    std::vector<cytnx_uint64> new_offj(shape.size());
    std::vector<cytnx_uint64> offj(shape.size());

    cytnx_uint64 accu = 1;
    for (cytnx_int32 i = shape.size() - 1; i >= 0; i--) {
      c_offj[i] = accu;
      accu *= shape[mapper[i]];
    }
    accu = 1;
    for (cytnx_int32 i = len.size() - 1; i >= 0; i--) {
      new_offj[i] = accu;
      accu *= len[i];

      // count-in the mapper:
      offj[i] = c_offj[mapper[i]];
    }

    if (this->device == Device.cpu) {
      utils_internal::uii.GetElems_ii[this->dtype](out->Mem, this->Mem, offj, new_offj, locators,
                                                   TotalElem);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      utils_internal::uii.cuGetElems_ii[this->dtype](out->Mem, this->Mem, offj, new_offj, locators,
                                                     TotalElem);
#else
      cytnx_error_msg(true, "[ERROR][GetElem_byShape] fatal internal%s",
                      "the Storage is set on gpu without CUDA support\n");
#endif
    }
  }

  void Storage_base::SetElem_byShape(boost::intrusive_ptr<Storage_base> &in,
                                     const std::vector<cytnx_uint64> &shape,
                                     const std::vector<cytnx_uint64> &mapper,
                                     const std::vector<cytnx_uint64> &len,
                                     const std::vector<std::vector<cytnx_uint64>> &locators,
                                     const bool &is_scalar) {
#ifdef UNI_DEBUG
    cytnx_error_msg(shape.size() != len.size(), "%s",
                    "[ERROR][DEBUG] internal Storage, shape.size() != len.size()");
#endif

    // std::cout <<"=====" << len.size() << " " << locators.size() << std::endl;
    // create new instance:
    cytnx_uint64 TotalElem = 1;
    for (cytnx_uint32 i = 0; i < len.size(); i++) TotalElem *= len[i];

    if (!is_scalar)
      cytnx_error_msg(in->size() != TotalElem, "%s",
                      "[ERROR] internal, the out Storage size does not match the no. of elems "
                      "calculated from Accessors.%s",
                      "\n");

    std::vector<cytnx_uint64> c_offj(shape.size());
    std::vector<cytnx_uint64> new_offj(shape.size());
    std::vector<cytnx_uint64> offj(shape.size());

    cytnx_uint64 accu = 1;
    for (cytnx_int32 i = shape.size() - 1; i >= 0; i--) {
      c_offj[i] = accu;
      accu *= shape[mapper[i]];
    }
    accu = 1;
    for (cytnx_int32 i = len.size() - 1; i >= 0; i--) {
      new_offj[i] = accu;
      accu *= len[i];

      // count-in the mapper:
      offj[i] = c_offj[mapper[i]];
    }

    if (this->device == Device.cpu) {
      if (utils_internal::uii.SetElems_ii[in->dtype][this->dtype] == NULL) {
        cytnx_error_msg(true, "%s", "[ERROR] %s",
                        "cannot assign complex element to real container.\n");
      }
      utils_internal::uii.SetElems_ii[in->dtype][this->dtype](in->Mem, this->Mem, offj, new_offj,
                                                              locators, TotalElem, is_scalar);
    } else {
#ifdef UNI_GPU
      if (utils_internal::uii.cuSetElems_ii[in->dtype][this->dtype] == NULL) {
        cytnx_error_msg(true, "%s", "[ERROR] %s",
                        "cannot assign complex element to real container.\n");
      }
      checkCudaErrors(cudaSetDevice(this->device));
      utils_internal::uii.cuSetElems_ii[in->dtype][this->dtype](in->Mem, this->Mem, offj, new_offj,
                                                                locators, TotalElem, is_scalar);
#else
      cytnx_error_msg(true, "[ERROR][SetElem_byShape] fatal internal%s",
                      "the Storage is set on gpu without CUDA support\n");
#endif
    }
  }

  // generators:
  void Storage_base::fill(const cytnx_complex128 &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_complex64 &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_double &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_float &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_int64 &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_uint64 &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_int32 &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_uint32 &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_int16 &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_uint16 &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_bool &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::set_zeros() {
    cytnx_error_msg(1, "%s", "[ERROR] call set_zeros directly on Void Storage.");
  }

  // instantiation:
  //================================================
  template <>
  float *Storage_base::data<float>() const {
    // check type
    cytnx_error_msg(dtype != Type.Float,
                    "[ERROR] type mismatch. try to get <float> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<float *>(this->Mem);
  }
  template <>
  double *Storage_base::data<double>() const {
    cytnx_error_msg(dtype != Type.Double,
                    "[ERROR] type mismatch. try to get <double> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<double *>(this->Mem);
  }

  template <>
  std::complex<double> *Storage_base::data<std::complex<double>>() const {
    cytnx_error_msg(
      dtype != Type.ComplexDouble,
      "[ERROR] type mismatch. try to get < complex<double> > type from raw data of type %s",
      Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cytnx_error_msg(this->device != Device.cpu, "%s",
                    "[ERROR] the Storage is on GPU but try to get with CUDA complex type "
                    "complex<double>. use type <cuDoubleComplex>  instead.");
    cudaDeviceSynchronize();
#endif
    return static_cast<std::complex<double> *>(this->Mem);
  }

  template <>
  std::complex<float> *Storage_base::data<std::complex<float>>() const {
    cytnx_error_msg(
      dtype != Type.ComplexFloat,
      "[ERROR] type mismatch. try to get < complex<float> > type from raw data of type %s",
      Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cytnx_error_msg(this->device != Device.cpu, "%s",
                    "[ERROR] the Storage is on GPU but try to get with CUDA complex type "
                    "complex<float>. use type <cuFloatComplex>  instead.");
    cudaDeviceSynchronize();
#endif
    return static_cast<std::complex<float> *>(this->Mem);
  }

  template <>
  uint32_t *Storage_base::data<uint32_t>() const {
    cytnx_error_msg(dtype != Type.Uint32,
                    "[ERROR] type mismatch. try to get <uint32_t> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<uint32_t *>(this->Mem);
  }

  template <>
  int32_t *Storage_base::data<int32_t>() const {
    cytnx_error_msg(dtype != Type.Int32,
                    "[ERROR] type mismatch. try to get <int32_t> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<int32_t *>(this->Mem);
  }

  template <>
  uint64_t *Storage_base::data<uint64_t>() const {
    cytnx_error_msg(dtype != Type.Uint64,
                    "[ERROR] type mismatch. try to get <uint64_t> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<uint64_t *>(this->Mem);
  }

  template <>
  int64_t *Storage_base::data<int64_t>() const {
    cytnx_error_msg(dtype != Type.Int64,
                    "[ERROR] type mismatch. try to get <int64_t> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<int64_t *>(this->Mem);
  }

  template <>
  int16_t *Storage_base::data<int16_t>() const {
    cytnx_error_msg(dtype != Type.Int16,
                    "[ERROR] type mismatch. try to get <int16_t> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<int16_t *>(this->Mem);
  }

  template <>
  uint16_t *Storage_base::data<uint16_t>() const {
    cytnx_error_msg(dtype != Type.Uint16,
                    "[ERROR] type mismatch. try to get <uint16_t> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<uint16_t *>(this->Mem);
  }

  template <>
  bool *Storage_base::data<bool>() const {
    cytnx_error_msg(dtype != Type.Bool,
                    "[ERROR] type mismatch. try to get <bool> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<bool *>(this->Mem);
  }

// get complex raw pointer using CUDA complex type
#ifdef UNI_GPU
  template <>
  cuDoubleComplex *Storage_base::data<cuDoubleComplex>() const {
    cytnx_error_msg(
      dtype != Type.ComplexDouble,
      "[ERROR] type mismatch. try to get <cuDoubleComplex> type from raw data of type %s",
      Type.getname(dtype).c_str());
    cytnx_error_msg(this->device == Device.cpu, "%s",
                    "[ERROR] the Storage is on CPU(Host) but try to get with CUDA complex type "
                    "cuDoubleComplex. use type <cytnx_complex128> or < complex<double> > instead.");
    cudaDeviceSynchronize();
    return static_cast<cuDoubleComplex *>(this->Mem);
  }
  template <>
  cuFloatComplex *Storage_base::data<cuFloatComplex>() const {
    cytnx_error_msg(
      dtype != Type.ComplexFloat,
      "[ERROR] type mismatch. try to get <cuFloatComplex> type from raw data of type %s",
      Type.getname(dtype).c_str());
    cytnx_error_msg(this->device == Device.cpu, "%s",
                    "[ERROR] the Storage is on CPU(Host) but try to get with CUDA complex type "
                    "cuFloatComplex. use type <cytnx_complex64> or < complex<float> > instead.");
    cudaDeviceSynchronize();
    return static_cast<cuFloatComplex *>(this->Mem);
  }
#endif

  // instantiation:
  //====================================================
  template <>
  float &Storage_base::at<float>(const unsigned int &idx) const {
    cytnx_error_msg(dtype != Type.Float,
                    "[ERROR] type mismatch. try to get <float> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<float *>(this->Mem)[idx];
  }
  template <>
  double &Storage_base::at<double>(const unsigned int &idx) const {
    cytnx_error_msg(dtype != Type.Double,
                    "[ERROR] type mismatch. try to get <double> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<double *>(this->Mem)[idx];
  }

  template <>
  std::complex<float> &Storage_base::at<std::complex<float>>(const unsigned int &idx) const {
    cytnx_error_msg(
      dtype != Type.ComplexFloat,
      "[ERROR] type mismatch. try to get < complex<float> > type from raw data of type %s",
      Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif

    return static_cast<complex<float> *>(this->Mem)[idx];
  }
  template <>
  std::complex<double> &Storage_base::at<std::complex<double>>(const unsigned int &idx) const {
    cytnx_error_msg(
      dtype != Type.ComplexDouble,
      "[ERROR] type mismatch. try to get < complex<double> > type from raw data of type %s",
      Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<complex<double> *>(this->Mem)[idx];
  }

  template <>
  uint32_t &Storage_base::at<uint32_t>(const unsigned int &idx) const {
    cytnx_error_msg(dtype != Type.Uint32,
                    "[ERROR] type mismatch. try to get <uint32_t> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<uint32_t *>(this->Mem)[idx];
  }

  template <>
  int32_t &Storage_base::at<int32_t>(const unsigned int &idx) const {
    cytnx_error_msg(dtype != Type.Int32,
                    "[ERROR] type mismatch. try to get <int32_t> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<int32_t *>(this->Mem)[idx];
  }

  template <>
  uint64_t &Storage_base::at<uint64_t>(const unsigned int &idx) const {
    cytnx_error_msg(dtype != Type.Uint64,
                    "[ERROR] type mismatch. try to get <uint64_t> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<uint64_t *>(this->Mem)[idx];
  }

  template <>
  int64_t &Storage_base::at<int64_t>(const unsigned int &idx) const {
    cytnx_error_msg(dtype != Type.Int64,
                    "[ERROR] type mismatch. try to get <int64_t> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<int64_t *>(this->Mem)[idx];
  }

  template <>
  uint16_t &Storage_base::at<uint16_t>(const unsigned int &idx) const {
    cytnx_error_msg(dtype != Type.Uint16,
                    "[ERROR] type mismatch. try to get <uint16_t> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<uint16_t *>(this->Mem)[idx];
  }

  template <>
  int16_t &Storage_base::at<int16_t>(const unsigned int &idx) const {
    cytnx_error_msg(dtype != Type.Int16,
                    "[ERROR] type mismatch. try to get <int16_t> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<int16_t *>(this->Mem)[idx];
  }

  template <>
  bool &Storage_base::at<bool>(const unsigned int &idx) const {
    cytnx_error_msg(dtype != Type.Bool,
                    "[ERROR] type mismatch. try to get <bool> type from raw data of type %s",
                    Type.getname(dtype).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<bool *>(this->Mem)[idx];
  }

}  // namespace cytnx
