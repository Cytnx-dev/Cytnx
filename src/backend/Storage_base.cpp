#include "backend/Storage.hpp"
#include "utils_internal_interface.hpp"
#include "utils/vec_print.hpp"

using namespace std;

namespace cytnx {

  // Storage Init interface.
  //=============================
  // boost::intrusive_ptr<Storage_base> SIInit_cd() {
  //   boost::intrusive_ptr<Storage_base> out(new ComplexDoubleStorage());
  //   return out;
  // }
  // boost::intrusive_ptr<Storage_base> SIInit_cf() {
  //   boost::intrusive_ptr<Storage_base> out(new ComplexFloatStorage());
  //   return out;
  // }
  // boost::intrusive_ptr<Storage_base> SIInit_d() {
  //   boost::intrusive_ptr<Storage_base> out(new DoubleStorage());
  //   return out;
  // }
  // boost::intrusive_ptr<Storage_base> SIInit_f() {
  //   boost::intrusive_ptr<Storage_base> out(new FloatStorage());
  //   return out;
  // }
  // boost::intrusive_ptr<Storage_base> SIInit_u64() {
  //   boost::intrusive_ptr<Storage_base> out(new Uint64Storage());
  //   return out;
  // }
  // boost::intrusive_ptr<Storage_base> SIInit_i64() {
  //   boost::intrusive_ptr<Storage_base> out(new Int64Storage());
  //   return out;
  // }
  // boost::intrusive_ptr<Storage_base> SIInit_u32() {
  //   boost::intrusive_ptr<Storage_base> out(new Uint32Storage());
  //   return out;
  // }
  // boost::intrusive_ptr<Storage_base> SIInit_i32() {
  //   boost::intrusive_ptr<Storage_base> out(new Int32Storage());
  //   return out;
  // }
  // boost::intrusive_ptr<Storage_base> SIInit_u16() {
  //   boost::intrusive_ptr<Storage_base> out(new Uint16Storage());
  //   return out;
  // }
  // boost::intrusive_ptr<Storage_base> SIInit_i16() {
  //   boost::intrusive_ptr<Storage_base> out(new Int16Storage());
  //   return out;
  // }
  // boost::intrusive_ptr<Storage_base> SIInit_b() {
  //   boost::intrusive_ptr<Storage_base> out(new BoolStorage());
  //   return out;
  // }

  // constexpr Storage_init_interface::Storage_init_interface() {
  //   USIInit.resize(N_Type);
  //   USIInit[this->Double] = SIInit_d;
  //   USIInit[this->Float] = SIInit_f;
  //   USIInit[this->ComplexDouble] = SIInit_cd;
  //   USIInit[this->ComplexFloat] = SIInit_cf;
  //   USIInit[this->Uint64] = SIInit_u64;
  //   USIInit[this->Int64] = SIInit_i64;
  //   USIInit[this->Uint32] = SIInit_u32;
  //   USIInit[this->Int32] = SIInit_i32;
  //   USIInit[this->Uint16] = SIInit_u16;
  //   USIInit[this->Int16] = SIInit_i16;
  //   USIInit[this->Bool] = SIInit_b;
  // }

  //==========================
  void Storage_base::Init(const unsigned long long &len_in, const int &device,
                          const bool &init_zero) {
    // cout << "Base.init" << endl;
  }

  Storage_base::Storage_base(const unsigned long long &len_in, const int &device,
                             const bool &init_zero) {
    this->Init(len_in, device, init_zero);
  }

  Storage_base &Storage_base::operator=(Storage_base &Rhs) {
    cout << "dev" << endl;
    return *this;
  }

  Storage_base::Storage_base(Storage_base &Rhs) { cout << "dev" << endl; }

  void Storage_base::resize(const cytnx_uint64 &newsize) {
    cytnx_error_msg(1, "[ERROR][internal] resize should not be called by base%s", "\n");
  }

  boost::intrusive_ptr<Storage_base> Storage_base::astype(const unsigned int &dtype) {
    boost::intrusive_ptr<Storage_base> out(new Storage_base());
    if (dtype == this->dtype()) return boost::intrusive_ptr<Storage_base>(this);

    if (this->device() == Device.cpu) {
      if (utils_internal::uii.ElemCast[this->dtype()][dtype] == NULL) {
        cytnx_error_msg(1, "[ERROR] not support type with dtype=%d", dtype);
      } else {
        utils_internal::uii.ElemCast[this->dtype()][dtype](this, out, this->size(), 1);
      }
    } else {
#ifdef UNI_GPU
      if (utils_internal::uii.cuElemCast[this->dtype()][dtype] == NULL) {
        cytnx_error_msg(1, "[ERROR] not support type with dtype=%d", dtype);
      } else {
        // std::cout << this->device() << std::endl;
        utils_internal::uii.cuElemCast[this->dtype()][dtype](this, out, this->size(),
                                                             this->device());
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
    return nullptr;
  }

  boost::intrusive_ptr<Storage_base> Storage_base::clone() {
    boost::intrusive_ptr<Storage_base> out(new Storage_base());
    return out;
  }

  string Storage_base::dtype_str() const { return Type.getname(this->dtype()); }
  string Storage_base::device_str() const { return Device.getname(this->device()); }
  void Storage_base::_Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device,
                                 const bool &iscap, const unsigned long long &cap_in) {
    cytnx_error_msg(1, "%s", "[ERROR] call _Init_byptr in base");
  }

  Storage_base::~Storage_base() {}

  void Storage_base::Move_memory_(const std::vector<cytnx_uint64> &old_shape,
                                  const std::vector<cytnx_uint64> &mapper,
                                  const std::vector<cytnx_uint64> &invmapper) {
    cytnx_error_msg(1, "%s", "[ERROR] call Move_memory_ directly on Void Storage.");
  }

  boost::intrusive_ptr<Storage_base> Storage_base::Move_memory(
    const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper,
    const std::vector<cytnx_uint64> &invmapper) {
    cytnx_error_msg(1, "%s", "[ERROR] call Move_memory_ directly on Void Storage.");
    return nullptr;
  }

  void Storage_base::to_(const int &device) {
    cytnx_error_msg(1, "%s", "[ERROR] call to_ directly on Void Storage.");
  }

  boost::intrusive_ptr<Storage_base> Storage_base::to(const int &device) {
    cytnx_error_msg(1, "%s", "[ERROR] call to directly on Void Storage.");
    return nullptr;
  }

  void Storage_base::PrintElem_byShape(std::ostream &os, const std::vector<cytnx_uint64> &shape,
                                       const std::vector<cytnx_uint64> &mapper) {
    cytnx_error_msg(1, "%s", "[ERROR] call PrintElem_byShape directly on Void Storage.");
  }

  void Storage_base::print_info() {
    cout << "dtype : " << this->dtype_str() << endl;
    cout << "device: " << Device.getname(this->device()) << endl;
    cout << "size  : " << this->size() << endl;
  }
  void Storage_base::print_elems() {
    cytnx_error_msg(1, "%s", "[ERROR] call print_elems directly on Void Storage.");
  }
  void Storage_base::print() {
    this->print_info();
    this->print_elems();
  }

  // shadow new:
  // [0] shape: shape of current TN
  // [x] direct feed-in accessor? accessor->.next() to get next index? no
  // we dont need mapper's information!
  // if len(locators) < shape.size(), it means last shape.size()-len(locators) axes are grouped.

  void Storage_base::GetElem_byShape_v2(boost::intrusive_ptr<Storage_base> &out,
                                        const std::vector<cytnx_uint64> &shape,
                                        const std::vector<std::vector<cytnx_uint64>> &locators,
                                        const cytnx_uint64 &Nunit) {
    if (User_debug)
      cytnx_error_msg(out->dtype() != this->dtype(), "%s", "[ERROR][DEBUG] %s",
                      "internal, the output dtype does not match current storage dtype.\n");

    cytnx_error_msg(this->device() != out->device(),
                    "[ERROR] cannot GetElem_byShape_v2 between different device.%s", "\n");
    cytnx_uint64 TotalElem = 1;
    for (cytnx_uint32 i = 0; i < locators.size(); i++) {
      if (locators[i].size())
        TotalElem *= locators[i].size();
      else  // axis get all!
        TotalElem *= shape[i];
    }

    // cytnx_error_msg(out->size() != TotalElem, "%s", "[ERROR] internal, the out Storage size does
    // not match the no. of elems calculated from Accessors.%s","\n");
    std::vector<cytnx_uint64> c_offj(locators.size());
    std::vector<cytnx_uint64> new_offj(locators.size());

    cytnx_uint64 caccu = 1, new_accu = 1;
    for (cytnx_int32 i = locators.size() - 1; i >= 0; i--) {
      c_offj[i] = caccu;
      caccu *= shape[i];

      new_offj[i] = new_accu;
      if (locators[i].size())
        new_accu *= locators[i].size();
      else
        new_accu *= shape[i];
    }
    // std::cout << c_offj << std::endl;
    // std::cout << new_offj << std::endl;
    // std::cout << TotalElem << std::endl;
    if (this->device() == Device.cpu) {
      utils_internal::uii.GetElems_conti_ii[this->dtype()](out->data(), this->data(), c_offj,
                                                           new_offj, locators, TotalElem, Nunit);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device()));
      // cytnx_error_msg(true,
      //                 "[Developing][GPU Getelem v2][Note, currently slice on GPU is disabled for
      //                 " "further inspection]%s",
      //                 "\n");
      utils_internal::uii.cuGetElems_conti_ii[this->dtype()](out->data(), this->data(), c_offj,
                                                             new_offj, locators, TotalElem, Nunit);
#else
      cytnx_error_msg(true, "[ERROR][GetElem_byShape] fatal internal%s",
                      "the Storage is set on gpu without CUDA support\n");
#endif
    }
  }

  void Storage_base::GetElem_byShape(boost::intrusive_ptr<Storage_base> &out,
                                     const std::vector<cytnx_uint64> &shape,
                                     const std::vector<cytnx_uint64> &mapper,
                                     const std::vector<cytnx_uint64> &len,
                                     const std::vector<std::vector<cytnx_uint64>> &locators) {
    if (User_debug) {
      cytnx_error_msg(shape.size() != len.size(), "%s",
                      "[ERROR][DEBUG] internal Storage, shape.size() != len.size()");
      cytnx_error_msg(out->dtype() != this->dtype(), "%s", "[ERROR][DEBUG] %s",
                      "internal, the output dtype does not match current storage dtype.\n");
    }
    cytnx_error_msg(this->device() != out->device(),
                    "[ERROR] cannot GetElem_byShape between different device.%s", "\n");

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

    if (this->device() == Device.cpu) {
      utils_internal::uii.GetElems_ii[this->dtype()](out->data(), this->data(), offj, new_offj,
                                                     locators, TotalElem);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device()));
      utils_internal::uii.cuGetElems_ii[this->dtype()](out->data(), this->data(), offj, new_offj,
                                                       locators, TotalElem);
#else
      cytnx_error_msg(true, "[ERROR][GetElem_byShape] fatal internal%s",
                      "the Storage is set on gpu without CUDA support\n");
#endif
    }
  }

  // This is deprecated !!
  void Storage_base::SetElem_byShape(boost::intrusive_ptr<Storage_base> &in,
                                     const std::vector<cytnx_uint64> &shape,
                                     const std::vector<cytnx_uint64> &mapper,
                                     const std::vector<cytnx_uint64> &len,
                                     const std::vector<std::vector<cytnx_uint64>> &locators,
                                     const bool &is_scalar) {
    if (User_debug)
      cytnx_error_msg(shape.size() != len.size(), "%s",
                      "[ERROR][DEBUG] internal Storage, shape.size() != len.size()");

    cytnx_error_msg(this->device() != in->device(),
                    "[ERROR] cannot SetElem_byShape between different device.%s", "\n");
    // std::cout <<"=====" << len.size() << " " << locators.size() << std::endl;
    // create new instance:
    cytnx_uint64 TotalElem = 1;
    for (cytnx_uint32 i = 0; i < len.size(); i++) TotalElem *= len[i];

    if (!is_scalar)
      cytnx_error_msg(in->size() != TotalElem, "%s",
                      "[ERROR] internal, the out Storage size does not match the no. of elems "
                      "calculated from Accessors.%s",
                      "\n");

    //[warning] this version only work for scalar currently!.

    std::vector<cytnx_uint64> c_offj(shape.size());
    std::vector<cytnx_uint64> new_offj(shape.size());
    std::vector<cytnx_uint64> offj(shape.size());

    cytnx_uint64 accu = 1;
    for (cytnx_int32 i = shape.size() - 1; i >= 0; i--) {
      c_offj[i] = accu;
      accu *= shape[i];
    }
    accu = 1;
    for (cytnx_int32 i = len.size() - 1; i >= 0; i--) {
      new_offj[i] = accu;
      accu *= len[i];

      // count-in the mapper:
      offj[i] = c_offj[mapper[i]];
    }

    if (this->device() == Device.cpu) {
      if (utils_internal::uii.SetElems_ii[in->dtype()][this->dtype()] == NULL) {
        cytnx_error_msg(true, "[ERROR] %s", "cannot assign complex element to real container.\n");
      }
      utils_internal::uii.SetElems_ii[in->dtype()][this->dtype()](
        in->data(), this->data(), c_offj, new_offj, locators, TotalElem, is_scalar);
    } else {
#ifdef UNI_GPU
      if (utils_internal::uii.cuSetElems_ii[in->dtype()][this->dtype()] == NULL) {
        cytnx_error_msg(true, "%s", "[ERROR] %s",
                        "cannot assign complex element to real container.\n");
      }
      checkCudaErrors(cudaSetDevice(this->device()));
      utils_internal::uii.cuSetElems_ii[in->dtype()][this->dtype()](
        in->data(), this->data(), offj, new_offj, locators, TotalElem, is_scalar);
#else
      cytnx_error_msg(true, "[ERROR][SetElem_byShape] fatal internal%s",
                      "the Storage is set on gpu without CUDA support\n");
#endif
    }
  }

  void Storage_base::SetElem_byShape_v2(boost::intrusive_ptr<Storage_base> &in,
                                        const std::vector<cytnx_uint64> &shape,
                                        const std::vector<std::vector<cytnx_uint64>> &locators,
                                        const cytnx_uint64 &Nunit, const bool &is_scalar) {
    // plan: we assume in is contiguous for now!
    //

    cytnx_error_msg(this->device() != in->device(),
                    "[ERROR] cannot SetElem_byShape_v2 between different device.%s", "\n");

    // std::cout <<"=====" << len.size() << " " << locators.size() << std::endl;
    // create new instance:
    cytnx_uint64 TotalElem = 1;
    for (cytnx_uint32 i = 0; i < locators.size(); i++) {
      if (locators[i].size())
        TotalElem *= locators[i].size();
      else  // axis get all!
        TotalElem *= shape[i];
    }

    // if(!is_scalar)
    //     cytnx_error_msg(in->size() != TotalElem, "%s", "[ERROR] internal, the out Storage size
    //     does not match the no. of elems calculated from Accessors.%s","\n");

    //[warning] this version only work for scalar currently!.

    std::vector<cytnx_uint64> c_offj(locators.size());
    std::vector<cytnx_uint64> new_offj(locators.size());

    cytnx_uint64 caccu = 1, new_accu = 1;
    for (cytnx_int32 i = locators.size() - 1; i >= 0; i--) {
      c_offj[i] = caccu;
      caccu *= shape[i];

      new_offj[i] = new_accu;
      if (locators[i].size())
        new_accu *= locators[i].size();
      else
        new_accu *= shape[i];
    }

    if (this->device() == Device.cpu) {
      if (utils_internal::uii.SetElems_conti_ii[in->dtype()][this->dtype()] == NULL) {
        cytnx_error_msg(true, "[ERROR] %s", "cannot assign complex element to real container.\n");
      }
      utils_internal::uii.SetElems_conti_ii[in->dtype()][this->dtype()](
        in->data(), this->data(), c_offj, new_offj, locators, TotalElem, Nunit, is_scalar);
    } else {
#ifdef UNI_GPU
      if (utils_internal::uii.cuSetElems_conti_ii[in->dtype()][this->dtype()] == NULL) {
        cytnx_error_msg(true, "[ERROR] %s", "cannot assign complex element to real container.\n");
      }
      checkCudaErrors(cudaSetDevice(this->device()));
      utils_internal::uii.cuSetElems_conti_ii[in->dtype()][this->dtype()](
        in->data(), this->data(), c_offj, new_offj, locators, TotalElem, Nunit, is_scalar);
      // cytnx_error_msg(true, "[Developing][SetElem on gpu is now down for further inspection]%s",
      //                 "\n");
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

  void Storage_base::append(const Scalar &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_complex128 &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_complex64 &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_double &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_float &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_int64 &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_uint64 &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_int32 &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_uint32 &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_int16 &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_uint16 &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_bool &val) {
    cytnx_error_msg(1, "%s", "[ERROR] call append directly on Void Storage.");
  }

  // instantiation:
  //================================================
  template <>
  float *Storage_base::data<float>() const {
    // check type
    cytnx_error_msg(this->dtype() != Type.Float,
                    "[ERROR] type mismatch. try to get <float> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<float *>(this->data());
  }
  template <>
  double *Storage_base::data<double>() const {
    cytnx_error_msg(this->dtype() != Type.Double,
                    "[ERROR] type mismatch. try to get <double> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<double *>(this->data());
  }

  template <>
  std::complex<double> *Storage_base::data<std::complex<double>>() const {
    cytnx_error_msg(
      this->dtype() != Type.ComplexDouble,
      "[ERROR] type mismatch. try to get < complex<double> > type from raw data of type %s",
      Type.getname(this->dtype()).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<std::complex<double> *>(this->data());
  }

  template <>
  std::complex<float> *Storage_base::data<std::complex<float>>() const {
    cytnx_error_msg(
      this->dtype() != Type.ComplexFloat,
      "[ERROR] type mismatch. try to get < complex<float> > type from raw data of type %s",
      Type.getname(this->dtype()).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<std::complex<float> *>(this->data());
  }

  template <>
  uint32_t *Storage_base::data<uint32_t>() const {
    cytnx_error_msg(this->dtype() != Type.Uint32,
                    "[ERROR] type mismatch. try to get <uint32_t> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<uint32_t *>(this->data());
  }

  template <>
  int32_t *Storage_base::data<int32_t>() const {
    cytnx_error_msg(this->dtype() != Type.Int32,
                    "[ERROR] type mismatch. try to get <int32_t> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<int32_t *>(this->data());
  }

  template <>
  uint64_t *Storage_base::data<uint64_t>() const {
    cytnx_error_msg(this->dtype() != Type.Uint64,
                    "[ERROR] type mismatch. try to get <uint64_t> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<uint64_t *>(this->data());
  }

  template <>
  int64_t *Storage_base::data<int64_t>() const {
    cytnx_error_msg(this->dtype() != Type.Int64,
                    "[ERROR] type mismatch. try to get <int64_t> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<int64_t *>(this->data());
  }

  template <>
  int16_t *Storage_base::data<int16_t>() const {
    cytnx_error_msg(this->dtype() != Type.Int16,
                    "[ERROR] type mismatch. try to get <int16_t> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<int16_t *>(this->data());
  }

  template <>
  uint16_t *Storage_base::data<uint16_t>() const {
    cytnx_error_msg(this->dtype() != Type.Uint16,
                    "[ERROR] type mismatch. try to get <uint16_t> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<uint16_t *>(this->data());
  }

  template <>
  bool *Storage_base::data<bool>() const {
    cytnx_error_msg(this->dtype() != Type.Bool,
                    "[ERROR] type mismatch. try to get <bool> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<bool *>(this->data());
  }

// get complex raw pointer using CUDA complex type
#ifdef UNI_GPU
  template <>
  cuDoubleComplex *Storage_base::data<cuDoubleComplex>() const {
    cytnx_error_msg(
      this->dtype() != Type.ComplexDouble,
      "[ERROR] type mismatch. try to get <cuDoubleComplex> type from raw data of type %s",
      Type.getname(this->dtype()).c_str());
    cytnx_error_msg(this->device() == Device.cpu, "%s",
                    "[ERROR] the Storage is on CPU(Host) but try to get with CUDA complex type "
                    "cuDoubleComplex. use type <cytnx_complex128> or < complex<double> > instead.");
    cudaDeviceSynchronize();
    return static_cast<cuDoubleComplex *>(this->data());
  }
  template <>
  cuFloatComplex *Storage_base::data<cuFloatComplex>() const {
    cytnx_error_msg(
      this->dtype() != Type.ComplexFloat,
      "[ERROR] type mismatch. try to get <cuFloatComplex> type from raw data of type %s",
      Type.getname(this->dtype()).c_str());
    cytnx_error_msg(this->device() == Device.cpu, "%s",
                    "[ERROR] the Storage is on CPU(Host) but try to get with CUDA complex type "
                    "cuFloatComplex. use type <cytnx_complex64> or < complex<float> > instead.");
    cudaDeviceSynchronize();
    return static_cast<cuFloatComplex *>(this->data());
  }
#endif

  // instantiation:
  //====================================================
  template <>
  float &Storage_base::at<float>(const cytnx_uint64 &idx) const {
    if (cytnx::User_debug) {
      cytnx_error_msg(this->dtype() != Type.Float,
                      "[ERROR] type mismatch. try to get <float> type from raw data of type %s",
                      Type.getname(this->dtype()).c_str());
    }
    if (idx >= this->size())
      cytnx_error_msg(1, "[ERROR] index [%d] out of bound [%d]\n", idx, this->size());

#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<float *>(this->data())[idx];
  }

  template <>
  double &Storage_base::at<double>(const cytnx_uint64 &idx) const {
    if (cytnx::User_debug) {
      cytnx_error_msg(this->dtype() != Type.Double,
                      "[ERROR] type mismatch. try to get <double> type from raw data of type %s",
                      Type.getname(this->dtype()).c_str());
    }
    if (idx >= this->size())
      cytnx_error_msg(1, "[ERROR] index [%d] out of bound [%d]\n", idx, this->size());

#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<double *>(this->data())[idx];
  }

  template <>
  std::complex<float> &Storage_base::at<std::complex<float>>(const cytnx_uint64 &idx) const {
    if (cytnx::User_debug)
      cytnx_error_msg(
        this->dtype() != Type.ComplexFloat,
        "[ERROR] type mismatch. try to get < complex<float> > type from raw data of type %s",
        Type.getname(this->dtype()).c_str());
    if (idx >= this->size())
      cytnx_error_msg(1, "[ERROR] index [%d] out of bound [%d]\n", idx, this->size());

#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<complex<float> *>(this->data())[idx];
  }

  template <>
  std::complex<double> &Storage_base::at<std::complex<double>>(const cytnx_uint64 &idx) const {
    if (cytnx::User_debug)
      cytnx_error_msg(
        this->dtype() != Type.ComplexDouble,
        "[ERROR] type mismatch. try to get < complex<double> > type from raw data of type %s",
        Type.getname(this->dtype()).c_str());
    if (idx >= this->size())
      cytnx_error_msg(1, "[ERROR] index [%d] out of bound [%d]\n", idx, this->size());

#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<complex<double> *>(this->data())[idx];
  }

  template <>
  uint32_t &Storage_base::at<uint32_t>(const cytnx_uint64 &idx) const {
    if (cytnx::User_debug)
      cytnx_error_msg(this->dtype() != Type.Uint32,
                      "[ERROR] type mismatch. try to get <uint32_t> type from raw data of type %s",
                      Type.getname(this->dtype()).c_str());
    if (idx >= this->size())
      cytnx_error_msg(1, "[ERROR] index [%d] out of bound [%d]\n", idx, this->size());

#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<uint32_t *>(this->data())[idx];
  }

  template <>
  int32_t &Storage_base::at<int32_t>(const cytnx_uint64 &idx) const {
    if (cytnx::User_debug)
      cytnx_error_msg(this->dtype() != Type.Int32,
                      "[ERROR] type mismatch. try to get <int32_t> type from raw data of type %s",
                      Type.getname(this->dtype()).c_str());
    if (idx >= this->size())
      cytnx_error_msg(1, "[ERROR] index [%d] out of bound [%d]\n", idx, this->size());

#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<int32_t *>(this->data())[idx];
  }

  template <>
  uint64_t &Storage_base::at<uint64_t>(const cytnx_uint64 &idx) const {
    if (cytnx::User_debug)
      cytnx_error_msg(this->dtype() != Type.Uint64,
                      "[ERROR] type mismatch. try to get <uint64_t> type from raw data of type %s",
                      Type.getname(this->dtype()).c_str());
    if (idx >= this->size())
      cytnx_error_msg(1, "[ERROR] index [%d] out of bound [%d]\n", idx, this->size());

#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<uint64_t *>(this->data())[idx];
  }

  template <>
  int64_t &Storage_base::at<int64_t>(const cytnx_uint64 &idx) const {
    if (cytnx::User_debug)
      cytnx_error_msg(this->dtype() != Type.Int64,
                      "[ERROR] type mismatch. try to get <int64_t> type from raw data of type %s",
                      Type.getname(this->dtype()).c_str());
    if (idx >= this->size())
      cytnx_error_msg(1, "[ERROR] index [%d] out of bound [%d]\n", idx, this->size());

#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<int64_t *>(this->data())[idx];
  }

  template <>
  uint16_t &Storage_base::at<uint16_t>(const cytnx_uint64 &idx) const {
    if (cytnx::User_debug)
      cytnx_error_msg(this->dtype() != Type.Uint16,
                      "[ERROR] type mismatch. try to get <uint16_t> type from raw data of type %s",
                      Type.getname(this->dtype()).c_str());

    if (idx >= this->size())
      cytnx_error_msg(1, "[ERROR] index [%d] out of bound [%d]\n", idx, this->size());

#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<uint16_t *>(this->data())[idx];
  }

  template <>
  int16_t &Storage_base::at<int16_t>(const cytnx_uint64 &idx) const {
    if (cytnx::User_debug)
      cytnx_error_msg(this->dtype() != Type.Int16,
                      "[ERROR] type mismatch. try to get <int16_t> type from raw data of type %s",
                      Type.getname(this->dtype()).c_str());
    if (idx >= this->size())
      cytnx_error_msg(1, "[ERROR] index [%d] out of bound [%d]\n", idx, this->size());

#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<int16_t *>(this->data())[idx];
  }

  template <>
  bool &Storage_base::at<bool>(const cytnx_uint64 &idx) const {
    if (cytnx::User_debug)
      cytnx_error_msg(this->dtype() != Type.Bool,
                      "[ERROR] type mismatch. try to get <bool> type from raw data of type %s",
                      Type.getname(this->dtype()).c_str());

    if (idx >= this->size())
      cytnx_error_msg(1, "[ERROR] index [%d] out of bound [%d]\n", idx, this->size());

#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<bool *>(this->data())[idx];
  }

  // instantiation:
  //====================================================
  template <>
  float &Storage_base::back<float>() const {
    cytnx_error_msg(this->dtype() != Type.Float,
                    "[ERROR] type mismatch. try to get <float> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
    cytnx_error_msg(this->size() == 0, "[ERROR] cannot call back on empty stoarge.%s", "\n");

#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<float *>(this->data())[this->size() - 1];
  }

  template <>
  double &Storage_base::back<double>() const {
    cytnx_error_msg(this->dtype() != Type.Double,
                    "[ERROR] type mismatch. try to get <double> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
    cytnx_error_msg(this->size() == 0, "[ERROR] cannot call back on empty stoarge.%s", "\n");
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<double *>(this->data())[this->size() - 1];
  }

  template <>
  std::complex<float> &Storage_base::back<std::complex<float>>() const {
    cytnx_error_msg(
      this->dtype() != Type.ComplexFloat,
      "[ERROR] type mismatch. try to get < complex<float> > type from raw data of type %s",
      Type.getname(this->dtype()).c_str());
    cytnx_error_msg(this->size() == 0, "[ERROR] cannot call back on empty stoarge.%s", "\n");
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif

    return static_cast<complex<float> *>(this->data())[this->size() - 1];
  }

  template <>
  std::complex<double> &Storage_base::back<std::complex<double>>() const {
    cytnx_error_msg(
      this->dtype() != Type.ComplexDouble,
      "[ERROR] type mismatch. try to get < complex<double> > type from raw data of type %s",
      Type.getname(this->dtype()).c_str());
    cytnx_error_msg(this->size() == 0, "[ERROR] cannot call back on empty stoarge.%s", "\n");
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<complex<double> *>(this->data())[this->size() - 1];
  }

  template <>
  uint32_t &Storage_base::back<uint32_t>() const {
    cytnx_error_msg(this->dtype() != Type.Uint32,
                    "[ERROR] type mismatch. try to get <uint32_t> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
    cytnx_error_msg(this->size() == 0, "[ERROR] cannot call back on empty stoarge.%s", "\n");
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<uint32_t *>(this->data())[this->size() - 1];
  }

  template <>
  int32_t &Storage_base::back<int32_t>() const {
    cytnx_error_msg(this->dtype() != Type.Int32,
                    "[ERROR] type mismatch. try to get <int32_t> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
    cytnx_error_msg(this->size() == 0, "[ERROR] cannot call back on empty stoarge.%s", "\n");
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<int32_t *>(this->data())[this->size() - 1];
  }

  template <>
  uint64_t &Storage_base::back<uint64_t>() const {
    cytnx_error_msg(this->dtype() != Type.Uint64,
                    "[ERROR] type mismatch. try to get <uint64_t> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
    cytnx_error_msg(this->size() == 0, "[ERROR] cannot call back on empty stoarge.%s", "\n");
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<uint64_t *>(this->data())[this->size() - 1];
  }

  template <>
  int64_t &Storage_base::back<int64_t>() const {
    cytnx_error_msg(this->dtype() != Type.Int64,
                    "[ERROR] type mismatch. try to get <int64_t> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
    cytnx_error_msg(this->size() == 0, "[ERROR] cannot call back on empty stoarge.%s", "\n");
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<int64_t *>(this->data())[this->size() - 1];
  }

  template <>
  uint16_t &Storage_base::back<uint16_t>() const {
    cytnx_error_msg(this->dtype() != Type.Uint16,
                    "[ERROR] type mismatch. try to get <uint16_t> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
    cytnx_error_msg(this->size() == 0, "[ERROR] cannot call back on empty stoarge.%s", "\n");
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<uint16_t *>(this->data())[this->size() - 1];
  }

  template <>
  int16_t &Storage_base::back<int16_t>() const {
    cytnx_error_msg(this->dtype() != Type.Int16,
                    "[ERROR] type mismatch. try to get <int16_t> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
    cytnx_error_msg(this->size() == 0, "[ERROR] cannot call back on empty stoarge.%s", "\n");
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<int16_t *>(this->data())[this->size() - 1];
  }

  template <>
  bool &Storage_base::back<bool>() const {
    cytnx_error_msg(this->dtype() != Type.Bool,
                    "[ERROR] type mismatch. try to get <bool> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
    cytnx_error_msg(this->size() == 0, "[ERROR] cannot call back on empty stoarge.%s", "\n");
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<bool *>(this->data())[this->size() - 1];
  }

  void Storage_base::_cpy_bool(void *ptr, const std::vector<cytnx_bool> &vin) {
    bool *tmp = static_cast<bool *>(ptr);

    for (cytnx_uint64 i = 0; i < vin.size(); i++) {
      tmp[i] = vin[i];
    }
  }

  boost::intrusive_ptr<Storage_base> Storage_base::real() {
    cytnx_error_msg(true, "[ERROR] trying to call Storage.real() from void Storage%s", "\n");
  }
  boost::intrusive_ptr<Storage_base> Storage_base::imag() {
    cytnx_error_msg(true, "[ERROR] trying to call Storage.imag() from void Storage%s", "\n");
  }

  Scalar Storage_base::get_item(const cytnx_uint64 &idx) const {
    cytnx_error_msg(true, "[ERROR] trying to call Storage.get_item() from void Storage%s", "\n");
    return Scalar();
  }

  void Storage_base::set_item(const cytnx_uint64 &idx, const Scalar &val) {
    cytnx_error_msg(true, "[ERROR] trying to call Storage.set_item() from void Storage%s", "\n");
  }
  void Storage_base::set_item(const cytnx_uint64 &idx, const cytnx_complex128 &val) {
    cytnx_error_msg(true, "[ERROR] trying to call Storage.set_item() from void Storage%s", "\n");
  }
  void Storage_base::set_item(const cytnx_uint64 &idx, const cytnx_complex64 &val) {
    cytnx_error_msg(true, "[ERROR] trying to call Storage.set_item() from void Storage%s", "\n");
  }
  void Storage_base::set_item(const cytnx_uint64 &idx, const cytnx_double &val) {
    cytnx_error_msg(true, "[ERROR] trying to call Storage.set_item() from void Storage%s", "\n");
  }
  void Storage_base::set_item(const cytnx_uint64 &idx, const cytnx_float &val) {
    cytnx_error_msg(true, "[ERROR] trying to call Storage.set_item() from void Storage%s", "\n");
  }
  void Storage_base::set_item(const cytnx_uint64 &idx, const cytnx_int64 &val) {
    cytnx_error_msg(true, "[ERROR] trying to call Storage.set_item() from void Storage%s", "\n");
  }
  void Storage_base::set_item(const cytnx_uint64 &idx, const cytnx_uint64 &val) {
    cytnx_error_msg(true, "[ERROR] trying to call Storage.set_item() from void Storage%s", "\n");
  }
  void Storage_base::set_item(const cytnx_uint64 &idx, const cytnx_int32 &val) {
    cytnx_error_msg(true, "[ERROR] trying to call Storage.set_item() from void Storage%s", "\n");
  }
  void Storage_base::set_item(const cytnx_uint64 &idx, const cytnx_uint32 &val) {
    cytnx_error_msg(true, "[ERROR] trying to call Storage.set_item() from void Storage%s", "\n");
  }
  void Storage_base::set_item(const cytnx_uint64 &idx, const cytnx_int16 &val) {
    cytnx_error_msg(true, "[ERROR] trying to call Storage.set_item() from void Storage%s", "\n");
  }
  void Storage_base::set_item(const cytnx_uint64 &idx, const cytnx_uint16 &val) {
    cytnx_error_msg(true, "[ERROR] trying to call Storage.set_item() from void Storage%s", "\n");
  }
  void Storage_base::set_item(const cytnx_uint64 &idx, const cytnx_bool &val) {
    cytnx_error_msg(true, "[ERROR] trying to call Storage.set_item() from void Storage%s", "\n");
  }

  // bool Storage_base::approx_eq(const boost::intrusive_ptr<Storage_base> &rhs,
  //                              const cytnx_double tol) {
  //   cytnx_error_msg(true, "[ERROR] trying to call Storage.approx_eq() from void Storage%s",
  //   "\n"); return false;
  // }

}  // namespace cytnx
