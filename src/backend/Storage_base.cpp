#include "backend/Storage.hpp"

#include "utils/vec_print.hpp"
#include "utils_internal_interface.hpp"

namespace cytnx {
  void Storage_base::Init(const unsigned long long &len_in, const int &device,
                          const bool &init_zero) {}

  Storage_base::Storage_base(const unsigned long long &len_in, const int &device,
                             const bool &init_zero) {
    this->Init(len_in, device, init_zero);
  }

  Storage_base &Storage_base::operator=(Storage_base &Rhs) {
    cytnx_error_msg(true, "[ERROR] Not implemented.%s", "\n");
    return *this;
  }

  Storage_base::Storage_base(Storage_base &Rhs) {
    cytnx_error_msg(true, "[ERROR] Not implemented.%s", "\n");
  }

  void Storage_base::resize(const cytnx_uint64 &newsize) {
    cytnx_error_msg(true, "[ERROR][internal] resize should not be called by base%s", "\n");
  }

  boost::intrusive_ptr<Storage_base> Storage_base::astype(const unsigned int &dtype) {
    boost::intrusive_ptr<Storage_base> out(new Storage_base());
    if (dtype == this->dtype()) return boost::intrusive_ptr<Storage_base>(this);

    if (this->device() == Device.cpu) {
      if (utils_internal::uii.ElemCast[this->dtype()][dtype] == nullptr) {
        cytnx_error_msg(true, "[ERROR] not support type with dtype=%d", dtype);
      } else {
        utils_internal::uii.ElemCast[this->dtype()][dtype](this, out, this->size(), 1);
      }
    } else {
#ifdef UNI_GPU
      if (utils_internal::uii.cuElemCast[this->dtype()][dtype] == nullptr) {
        cytnx_error_msg(true, "[ERROR] not support type with dtype=%d", dtype);
      } else {
        utils_internal::uii.cuElemCast[this->dtype()][dtype](this, out, this->size(),
                                                             this->device());
      }
#else
      cytnx_error_msg(
        true, "%s",
        "[ERROR][Internal Error] enter GPU section without CUDA support @ Storage.astype()");
#endif
    }
    return out;
  }

  boost::intrusive_ptr<Storage_base> Storage_base::_create_new_sametype() {
    cytnx_error_msg(true, "%s", "[ERROR] call _create_new_sametype in base");
    return nullptr;
  }

  boost::intrusive_ptr<Storage_base> Storage_base::clone() {
    boost::intrusive_ptr<Storage_base> out(new Storage_base());
    return out;
  }

  std::string Storage_base::dtype_str() const { return Type.getname(this->dtype()); }
  std::string Storage_base::device_str() const { return Device.getname(this->device()); }
  void Storage_base::_Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device,
                                 const bool &iscap, const unsigned long long &cap_in) {
    cytnx_error_msg(true, "%s", "[ERROR] call _Init_byptr in base");
  }

  Storage_base::~Storage_base() {}

  void Storage_base::Move_memory_(const std::vector<cytnx_uint64> &old_shape,
                                  const std::vector<cytnx_uint64> &mapper,
                                  const std::vector<cytnx_uint64> &invmapper) {
    cytnx_error_msg(true, "%s", "[ERROR] call Move_memory_ directly on Void Storage.");
  }

  boost::intrusive_ptr<Storage_base> Storage_base::Move_memory(
    const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper,
    const std::vector<cytnx_uint64> &invmapper) {
    cytnx_error_msg(true, "%s", "[ERROR] call Move_memory_ directly on Void Storage.");
    return nullptr;
  }

  void Storage_base::to_(const int &device) {
    cytnx_error_msg(true, "%s", "[ERROR] call to_ directly on Void Storage.");
  }

  boost::intrusive_ptr<Storage_base> Storage_base::to(const int &device) {
    cytnx_error_msg(true, "%s", "[ERROR] call to directly on Void Storage.");
    return nullptr;
  }

  void Storage_base::PrintElem_byShape(std::ostream &os, const std::vector<cytnx_uint64> &shape,
                                       const std::vector<cytnx_uint64> &mapper) {
    cytnx_error_msg(true, "%s", "[ERROR] call PrintElem_byShape directly on Void Storage.");
  }

  void Storage_base::print_info() {
    std::cout << "dtype : " << this->dtype_str() << std::endl;
    std::cout << "device: " << Device.getname(this->device()) << std::endl;
    std::cout << "size  : " << this->size() << std::endl;
  }
  void Storage_base::print_elems() {
    cytnx_error_msg(true, "%s", "[ERROR] call print_elems directly on Void Storage.");
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
                    "[ERROR] Cannot GetElem_byShape_v2 between different device.%s", "\n");
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
                    "[ERROR] Cannot GetElem_byShape between different device.%s", "\n");
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
                    "[ERROR] Cannot SetElem_byShape between different device.%s", "\n");
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
      if (utils_internal::uii.SetElems_ii[in->dtype()][this->dtype()] == nullptr) {
        cytnx_error_msg(true, "[ERROR] %s", "cannot assign complex element to real container.\n");
      }
      utils_internal::uii.SetElems_ii[in->dtype()][this->dtype()](
        in->data(), this->data(), c_offj, new_offj, locators, TotalElem, is_scalar);
    } else {
#ifdef UNI_GPU
      if (utils_internal::uii.cuSetElems_ii[in->dtype()][this->dtype()] == nullptr) {
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
    cytnx_error_msg(this->device() != in->device(),
                    "[ERROR] Cannot SetElem_byShape_v2 between different device.%s", "\n");
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
      if (utils_internal::uii.SetElems_conti_ii[in->dtype()][this->dtype()] == nullptr) {
        cytnx_error_msg(true, "[ERROR] %s", "cannot assign complex element to real container.\n");
      }
      utils_internal::uii.SetElems_conti_ii[in->dtype()][this->dtype()](
        in->data(), this->data(), c_offj, new_offj, locators, TotalElem, Nunit, is_scalar);
    } else {
#ifdef UNI_GPU
      if (utils_internal::uii.cuSetElems_conti_ii[in->dtype()][this->dtype()] == nullptr) {
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
    cytnx_error_msg(true, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_complex64 &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_double &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_float &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_int64 &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_uint64 &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_int32 &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_uint32 &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_int16 &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_uint16 &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::fill(const cytnx_bool &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call fill directly on Void Storage.");
  }
  void Storage_base::set_zeros() {
    cytnx_error_msg(true, "%s", "[ERROR] call set_zeros directly on Void Storage.");
  }

  void Storage_base::append(const Scalar &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_complex128 &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_complex64 &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_double &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_float &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_int64 &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_uint64 &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_int32 &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_uint32 &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_int16 &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_uint16 &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call append directly on Void Storage.");
  }
  void Storage_base::append(const cytnx_bool &val) {
    cytnx_error_msg(true, "%s", "[ERROR] call append directly on Void Storage.");
  }

  // instantiation:
  //================================================
  namespace {
    // C++ type spelling embedded in the at/back/data dtype-mismatch messages
    // (e.g. "<float>", "< std::complex<double> >").
    //
    // The primary template is ill-formed if instantiated: every type reaching those
    // messages must have an explicit specialization below. Adding a new at/back/data
    // instantiation without a matching spelling is therefore a compile error, rather
    // than silently feeding a null pointer to the "%s" conversion (runtime UB).
    template <typename T>
    struct type_spelling_holder {
      static_assert(always_false_v<T>, "type_spelling is not specialized for this type");
      static constexpr const char *value = nullptr;
    };
    template <typename T>
    constexpr const char *type_spelling = type_spelling_holder<T>::value;
    template <>
    constexpr const char *type_spelling<float> = "<float>";
    template <>
    constexpr const char *type_spelling<double> = "<double>";
    template <>
    constexpr const char *type_spelling<std::complex<double>> = "< std::complex<double> >";
    template <>
    constexpr const char *type_spelling<std::complex<float>> = "< std::complex<float> >";
    template <>
    constexpr const char *type_spelling<uint32_t> = "<uint32_t>";
    template <>
    constexpr const char *type_spelling<int32_t> = "<int32_t>";
    template <>
    constexpr const char *type_spelling<uint64_t> = "<uint64_t>";
    template <>
    constexpr const char *type_spelling<int64_t> = "<int64_t>";
    template <>
    constexpr const char *type_spelling<uint16_t> = "<uint16_t>";
    template <>
    constexpr const char *type_spelling<int16_t> = "<int16_t>";
    template <>
    constexpr const char *type_spelling<bool> = "<bool>";
  }  // namespace

  template <CytnxType T>
  T *Storage_base::data() const {
    // check type
    cytnx_error_msg(this->dtype() != Type_class::cy_typeid_v<T>,
                    "[ERROR] type mismatch. try to get %s type from raw data of type %s",
                    type_spelling<T>, Type.getname(this->dtype()).c_str());
#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<T *>(this->data());
  }

  template float *Storage_base::data<float>() const;
  template double *Storage_base::data<double>() const;
  template std::complex<double> *Storage_base::data<std::complex<double>>() const;
  template std::complex<float> *Storage_base::data<std::complex<float>>() const;
  template uint32_t *Storage_base::data<uint32_t>() const;
  template int32_t *Storage_base::data<int32_t>() const;
  template uint64_t *Storage_base::data<uint64_t>() const;
  template int64_t *Storage_base::data<int64_t>() const;
  template int16_t *Storage_base::data<int16_t>() const;
  template uint16_t *Storage_base::data<uint16_t>() const;
  template bool *Storage_base::data<bool>() const;

// get complex raw pointer using CUDA complex type
#ifdef UNI_GPU
  template <>
  cuDoubleComplex *Storage_base::data<cuDoubleComplex>() const {
    cytnx_error_msg(
      this->dtype() != Type.ComplexDouble,
      "[ERROR] type mismatch. try to get <cuDoubleComplex> type from raw data of type %s",
      Type.getname(this->dtype()).c_str());
    cytnx_error_msg(
      this->device() == Device.cpu, "%s",
      "[ERROR] the Storage is on CPU(Host) but try to get with CUDA complex type "
      "cuDoubleComplex. use type <cytnx_complex128> or < std::complex<double> > instead.");
    cudaDeviceSynchronize();
    return static_cast<cuDoubleComplex *>(this->data());
  }
  template <>
  cuFloatComplex *Storage_base::data<cuFloatComplex>() const {
    cytnx_error_msg(
      this->dtype() != Type.ComplexFloat,
      "[ERROR] type mismatch. try to get <cuFloatComplex> type from raw data of type %s",
      Type.getname(this->dtype()).c_str());
    cytnx_error_msg(
      this->device() == Device.cpu, "%s",
      "[ERROR] the Storage is on CPU(Host) but try to get with CUDA complex type "
      "cuFloatComplex. use type <cytnx_complex64> or < std::complex<float> > instead.");
    cudaDeviceSynchronize();
    return static_cast<cuFloatComplex *>(this->data());
  }
  // cuda::std::complex views -- the representation GPU kernels use internally (#1004).
  template <>
  cytnx_cuda_complex128 *Storage_base::data<cytnx_cuda_complex128>() const {
    cytnx_error_msg(
      this->dtype() != Type.ComplexDouble,
      "[ERROR] type mismatch. try to get < cuda::std::complex<double> > type from raw "
      "data of type %s",
      Type.getname(this->dtype()).c_str());
    cytnx_error_msg(this->device() == Device.cpu, "%s",
                    "[ERROR] the Storage is on CPU(Host) but try to get with CUDA complex type "
                    "cuda::std::complex<double>. use type <cytnx_complex128> or < "
                    "std::complex<double> > instead.");
    cudaDeviceSynchronize();
    return static_cast<cytnx_cuda_complex128 *>(this->data());
  }
  template <>
  cytnx_cuda_complex64 *Storage_base::data<cytnx_cuda_complex64>() const {
    cytnx_error_msg(this->dtype() != Type.ComplexFloat,
                    "[ERROR] type mismatch. try to get < cuda::std::complex<float> > type from raw "
                    "data of type %s",
                    Type.getname(this->dtype()).c_str());
    cytnx_error_msg(
      this->device() == Device.cpu, "%s",
      "[ERROR] the Storage is on CPU(Host) but try to get with CUDA complex type "
      "cuda::std::complex<float>. use type <cytnx_complex64> or < std::complex<float> "
      "> instead.");
    cudaDeviceSynchronize();
    return static_cast<cytnx_cuda_complex64 *>(this->data());
  }
#endif

  // instantiation:
  //====================================================
  template <CytnxType T>
  T &Storage_base::at(const cytnx_uint64 &idx) const {
    cytnx_error_msg(this->dtype() != Type_class::cy_typeid_v<T>,
                    "[ERROR] type mismatch. try to get %s type from raw data of type %s",
                    type_spelling<T>, Type.getname(this->dtype()).c_str());
    if (idx >= this->size())
      cytnx_error_msg(true, "[ERROR] index [%llu] out of bound [%llu]\n",
                      static_cast<unsigned long long>(idx),
                      static_cast<unsigned long long>(this->size()));

#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<T *>(this->data())[idx];
  }

  template float &Storage_base::at<float>(const cytnx_uint64 &idx) const;
  template double &Storage_base::at<double>(const cytnx_uint64 &idx) const;
  template std::complex<float> &Storage_base::at<std::complex<float>>(
    const cytnx_uint64 &idx) const;
  template std::complex<double> &Storage_base::at<std::complex<double>>(
    const cytnx_uint64 &idx) const;
  template uint32_t &Storage_base::at<uint32_t>(const cytnx_uint64 &idx) const;
  template int32_t &Storage_base::at<int32_t>(const cytnx_uint64 &idx) const;
  template uint64_t &Storage_base::at<uint64_t>(const cytnx_uint64 &idx) const;
  template int64_t &Storage_base::at<int64_t>(const cytnx_uint64 &idx) const;
  template uint16_t &Storage_base::at<uint16_t>(const cytnx_uint64 &idx) const;
  template int16_t &Storage_base::at<int16_t>(const cytnx_uint64 &idx) const;
  template bool &Storage_base::at<bool>(const cytnx_uint64 &idx) const;

  // instantiation:
  //====================================================
  template <CytnxType T>
  T &Storage_base::back() const {
    cytnx_error_msg(this->dtype() != Type_class::cy_typeid_v<T>,
                    "[ERROR] type mismatch. try to get %s type from raw data of type %s",
                    type_spelling<T>, Type.getname(this->dtype()).c_str());
    cytnx_error_msg(this->size() == 0, "[ERROR] Cannot call back on empty stoarge.%s", "\n");

#ifdef UNI_GPU
    cudaDeviceSynchronize();
#endif
    return static_cast<T *>(this->data())[this->size() - 1];
  }

  template float &Storage_base::back<float>() const;
  template double &Storage_base::back<double>() const;
  template std::complex<float> &Storage_base::back<std::complex<float>>() const;
  template std::complex<double> &Storage_base::back<std::complex<double>>() const;
  template uint32_t &Storage_base::back<uint32_t>() const;
  template int32_t &Storage_base::back<int32_t>() const;
  template uint64_t &Storage_base::back<uint64_t>() const;
  template int64_t &Storage_base::back<int64_t>() const;
  template uint16_t &Storage_base::back<uint16_t>() const;
  template int16_t &Storage_base::back<int16_t>() const;
  template bool &Storage_base::back<bool>() const;

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
