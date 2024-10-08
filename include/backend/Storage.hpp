#ifndef BACKEND_STORAGE_H_
#define BACKEND_STORAGE_H_
#ifndef BACKEND_TORCH

  #include <algorithm>
  #include <complex>
  #include <cstdlib>
  #include <cstdio>
  #include <cstring>
  #include <fstream>
  #include <initializer_list>
  #include <iostream>
  #include <memory>
  #include <string>
  #include <type_traits>
  #include <typeinfo>
  #include <utility>
  #include <vector>

  #include "backend/Scalar.hpp"
  #include "backend/utils_internal_gpu/cuAlloc_gpu.hpp"
  #include "backend/utils_internal_gpu/cuComplexmem_gpu.hpp"
  #include "backend/utils_internal_gpu/cuFill_gpu.hpp"
  #include "backend/utils_internal_gpu/cuMovemem_gpu.hpp"
  #include "cytnx_error.hpp"
  #include "Device.hpp"
  #include "intrusive_ptr_base.hpp"
  #include "Type.hpp"

  #define STORAGE_DEFT_SZ 2

namespace cytnx {

  ///@cond
  class Storage_base : public intrusive_ptr_base<Storage_base> {
   public:
    void *Mem;
    // std::vector<unsigned int> shape;

    unsigned long long len;  // default 0
    unsigned long long cap;  // default 0
    unsigned int dtype;  // default 0, Void
    int device;  // default -1, on cpu

    Storage_base() : cap(0), len(0), Mem(NULL), dtype(0), device(-1){};
    // Storage_base(const std::initializer_list<unsigned int> &init_shape);
    // Storage_base(const std::vector<unsigned int> &init_shape);
    Storage_base(const unsigned long long &len_in, const int &device, const bool &init_zero = true);

    Storage_base(Storage_base &Rhs);
    Storage_base &operator=(Storage_base &Rhs);
    boost::intrusive_ptr<Storage_base> astype(const unsigned int &dtype);

    // void Init(const std::initializer_list<unsigned int> &init_shape);
    std::string dtype_str() const;
    std::string device_str() const;
    const unsigned long long &capacity() const { return this->cap; }
    const unsigned long long &size() const { return this->len; }
    virtual ~Storage_base();

    template <class T>
    T &at(const cytnx_uint64 &idx) const;

    template <class T>
    T &back() const;

    template <class T>
    T *data() const;

    void *data() const { return this->Mem; }

    void _cpy_bool(void *ptr, const std::vector<cytnx_bool> &vin);

    void print();
    void print_info();
    /*
        This function is design to check the type mismatch.
        Handy for developer to exclude the assign of double
        C pointer into a non-DoubleStorage.

        For example:
        float *cptr = (float*)calloc(4,sizeof(float));

        intrusive_ptr<Storage> array(new DoubleStorage());
        array->_Init_byptr((void*)cptr,4); // This is fatal, since we alloc cptr as float,
                                           // but apon free, DoubleStorage will free 2x
                                           // of memory!!!!

        array->_Init_byptr_safe(cptr,4);   // This is design to avoid the above problem
                                           // by checking the type of input pointer with
                                           // the type of Storage before call _Init_byptr.
                                           // [Note] this will intorduce overhead!!.

    */
    template <class T>
    void _Init_byptr_safe(T *rawptr, const unsigned long long &len_in) {
      // check:
      if (this->dtype == Type.Float) {
        cytnx_error_msg(typeid(T) != typeid(cytnx_float), "%s",
                        "[ERROR _Init_byptr_safe type not match]");
      } else if (this->dtype == Type.Double) {
        cytnx_error_msg(typeid(T) != typeid(cytnx_double), "%s",
                        "[ERROR _Init_byptr_safe type not match]");
      } else if (this->dtype == Type.Uint64) {
        cytnx_error_msg(typeid(T) != typeid(cytnx_uint64), "%s",
                        "[ERROR _Init_byptr_safe type not match]");
      } else if (this->dtype == Type.Uint32) {
        cytnx_error_msg(typeid(T) != typeid(cytnx_uint32), "%s",
                        "[ERROR _Init_byptr_safe type not match]");
      } else if (this->dtype == Type.Int64) {
        cytnx_error_msg(typeid(T) != typeid(cytnx_int64), "%s",
                        "[ERROR _Init_byptr_safe type not match]");
      } else if (this->dtype == Type.Int32) {
        cytnx_error_msg(typeid(T) != typeid(cytnx_int32), "%s",
                        "[ERROR _Init_byptr_safe type not match]");
      } else if (this->dtype == Type.ComplexDouble) {
        cytnx_error_msg(typeid(T) != typeid(cytnx_complex128), "%s",
                        "[ERROR _Init_byptr_safe type not match]");
      } else if (this->dtype == Type.ComplexFloat) {
        cytnx_error_msg(typeid(T) != typeid(cytnx_complex64), "%s",
                        "[ERROR _Init_byptr_safe type not match]");
      } else if (this->dtype == Type.Int16) {
        cytnx_error_msg(typeid(T) != typeid(cytnx_int16), "%s",
                        "[ERROR _Init_byptr_safe type not match]");
      } else if (this->dtype == Type.Uint16) {
        cytnx_error_msg(typeid(T) != typeid(cytnx_uint16), "%s",
                        "[ERROR _Init_byptr_safe type not match]");
      } else if (this->dtype == Type.Bool) {
        cytnx_error_msg(typeid(T) != typeid(cytnx_bool), "%s",
                        "[ERROR _Init_byptr_safe type not match]");
      } else {
        cytnx_error_msg(1, "[FATAL] ERROR%s", "\n");
      }

      this->_Init_byptr((void *)rawptr, len_in);
    }

    void GetElem_byShape_v2(boost::intrusive_ptr<Storage_base> &out,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &Nunit);
    void GetElem_byShape(boost::intrusive_ptr<Storage_base> &out,
                         const std::vector<cytnx_uint64> &shape,
                         const std::vector<cytnx_uint64> &mapper,
                         const std::vector<cytnx_uint64> &len,
                         const std::vector<std::vector<cytnx_uint64>> &locators);
    void SetElem_byShape(boost::intrusive_ptr<Storage_base> &in,
                         const std::vector<cytnx_uint64> &shape,
                         const std::vector<cytnx_uint64> &mapper,
                         const std::vector<cytnx_uint64> &len,
                         const std::vector<std::vector<cytnx_uint64>> &locators,
                         const bool &is_scalar);
    void SetElem_byShape_v2(boost::intrusive_ptr<Storage_base> &in,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &Nunit, const bool &is_scalar);
    // these is the one that do the work, and customize with Storage_base
    // virtual void Init(const std::vector<unsigned int> &init_shape);
    virtual void Init(const unsigned long long &len_in, const int &device = -1,
                      const bool &init_zero = true);
    virtual void _Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device = -1,
                             const bool &iscap = false, const unsigned long long &cap_in = 0);

    // this function will return a new storage with the same type as the one
    // that initiate this function.
    virtual boost::intrusive_ptr<Storage_base> _create_new_sametype();

    // [future] this will move the memory to device / cpu
    virtual void to_(const int &device);
    virtual boost::intrusive_ptr<Storage_base> to(const int &device);

    virtual boost::intrusive_ptr<Storage_base> clone();

    // this will perform permute on the underlying memory.
    virtual boost::intrusive_ptr<Storage_base> Move_memory(
      const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper,
      const std::vector<cytnx_uint64> &invmapper);
    virtual void Move_memory_(const std::vector<cytnx_uint64> &old_shape,
                              const std::vector<cytnx_uint64> &mapper,
                              const std::vector<cytnx_uint64> &invmapper);
    virtual void PrintElem_byShape(std::ostream &os, const std::vector<cytnx_uint64> &shape,
                                   const std::vector<cytnx_uint64> &mapper = {});
    virtual void print_elems();

    // TODO: only for complex storage
    virtual boost::intrusive_ptr<Storage_base> real();
    virtual boost::intrusive_ptr<Storage_base> imag();

    // generators:
    virtual void fill(const cytnx_complex128 &val);
    virtual void fill(const cytnx_complex64 &val);
    virtual void fill(const cytnx_double &val);
    virtual void fill(const cytnx_float &val);
    virtual void fill(const cytnx_int64 &val);
    virtual void fill(const cytnx_uint64 &val);
    virtual void fill(const cytnx_int32 &val);
    virtual void fill(const cytnx_uint32 &val);
    virtual void fill(const cytnx_int16 &val);
    virtual void fill(const cytnx_uint16 &val);
    virtual void fill(const cytnx_bool &val);
    virtual void set_zeros();
    virtual void resize(const cytnx_uint64 &newsize);

    virtual void append(const Scalar &val);
    virtual void append(const cytnx_complex128 &val);
    virtual void append(const cytnx_complex64 &val);
    virtual void append(const cytnx_double &val);
    virtual void append(const cytnx_float &val);
    virtual void append(const cytnx_int64 &val);
    virtual void append(const cytnx_uint64 &val);
    virtual void append(const cytnx_int32 &val);
    virtual void append(const cytnx_uint32 &val);
    virtual void append(const cytnx_int16 &val);
    virtual void append(const cytnx_uint16 &val);
    virtual void append(const cytnx_bool &val);

    virtual Scalar get_item(const cytnx_uint64 &in) const;

    virtual void set_item(const cytnx_uint64 &idx, const Scalar &val);
    virtual void set_item(const cytnx_uint64 &idx, const cytnx_complex128 &val);
    virtual void set_item(const cytnx_uint64 &idx, const cytnx_complex64 &val);
    virtual void set_item(const cytnx_uint64 &idx, const cytnx_double &val);
    virtual void set_item(const cytnx_uint64 &idx, const cytnx_float &val);
    virtual void set_item(const cytnx_uint64 &idx, const cytnx_int64 &val);
    virtual void set_item(const cytnx_uint64 &idx, const cytnx_uint64 &val);
    virtual void set_item(const cytnx_uint64 &idx, const cytnx_int32 &val);
    virtual void set_item(const cytnx_uint64 &idx, const cytnx_uint32 &val);
    virtual void set_item(const cytnx_uint64 &idx, const cytnx_int16 &val);
    virtual void set_item(const cytnx_uint64 &idx, const cytnx_uint16 &val);
    virtual void set_item(const cytnx_uint64 &idx, const cytnx_bool &val);

    // virtual bool approx_eq(const boost::intrusive_ptr<Storage_base> &rhs,
    //                        const cytnx_double tol = 1e-8);
  };
    ///@endcond

  #ifdef UNI_GPU
  template <typename T>
  class GpuAllocator {
   public:
    typedef T value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    [[nodiscard]] constexpr T *allocate(std::size_t n) {
      return reinterpret_cast<T *>(utils_internal::cuMalloc_gpu(n * sizeof(T)));
    }

    constexpr void deallocate(T *p, [[maybe_unused]] std::size_t n) {
      checkCudaErrors(cudaFree(p));
    }

    template <typename Alloc>
    bool operator==(const Alloc &rhs) const noexcept {
      return std::is_same_v<GpuAllocator<T>, Alloc>;
    }

    template <typename Alloc>
    bool operator!=(const Alloc &rhs) const noexcept {
      return !operator==(rhs);
    }
  };
  #else
  template <typename T>
  class GpuAllocator {};
  #endif  // UNI_GPU

  template <class T, auto v>
  struct TypePair {
    using type = T;
    static constexpr auto value = v;
  };

  template <class T, class... Bn>
  struct FindTypePair {};
  template <class T, class B1, class... Bn>
  struct FindTypePair<T, B1, Bn...>
      : std::conditional_t<std::is_same_v<T, typename B1::type>, B1, FindTypePair<T, Bn...>> {};

  template <class T>
  inline constexpr auto cytnx_type_id = FindTypePair<
    T, TypePair<cytnx_complex128, Type_class::ComplexDouble>,
    TypePair<cytnx_complex64, Type_class::ComplexFloat>, TypePair<cytnx_double, Type_class::Double>,
    TypePair<cytnx_float, Type_class::Float>, TypePair<cytnx_uint64, Type_class::Uint64>,
    TypePair<cytnx_int64, Type_class::Int64>, TypePair<cytnx_uint32, Type_class::Uint32>,
    TypePair<cytnx_int32, Type_class::Int32>, TypePair<cytnx_uint16, Type_class::Uint16>,
    TypePair<cytnx_int16, Type_class::Int16>, TypePair<char, Type_class::Bool>>::value;

  template <typename T, typename Allocator = std::allocator<T>>
  class StorageImplementation : public Storage_base {
   public:
    typedef T value_type;
    typedef Allocator allocator_type;
    typedef std::vector<T, Allocator> storage_type;
    typedef typename storage_type::size_type size_type;
    typedef typename storage_type::iterator iterator;
    typedef typename storage_type::const_iterator const_iterator;

    static_assert(
      std::is_same_v<value_type, cytnx_complex128> || std::is_same_v<value_type, cytnx_complex64> ||
      std::is_same_v<value_type, cytnx_double> || std::is_same_v<value_type, cytnx_float> ||
      std::is_same_v<value_type, cytnx_int64> || std::is_same_v<value_type, cytnx_uint64> ||
      std::is_same_v<value_type, cytnx_int32> || std::is_same_v<value_type, cytnx_uint32> ||
      std::is_same_v<value_type, cytnx_int16> || std::is_same_v<value_type, cytnx_uint16> ||
      std::is_same_v<value_type, char>);

    StorageImplementation() {
      // TODO: initialize it in Storage_base
      dtype = cytnx_type_id<T>;
    }

    StorageImplementation(int device) : StorageImplementation() {
      // TODO: initialize it in Storage_base
      this->device = device;
    }

    StorageImplementation(unsigned long long len_in, int device = -1, bool init_zero = true)
        : StorageImplementation(device) {
      Init(len_in, device, init_zero);
    }
    StorageImplementation(const StorageImplementation &other) {
      dtype = cytnx_type_id<T>;
      // To make sure `cap` equals storage_.capacity() and only allocates the memory once, we have
      // to reserve and then assign.
      cap = other.cap;
      storage_.reserve(cap);
      storage_ = other.storage_;
      len = storage_.size();
      Mem = reinterpret_cast<void *>(storage_.data());
      device = other.device;
    }

    void Init(const unsigned long long &len_in, const int &device = -1,
              const bool &init_zero = true) override {
      len = len_in;
      if (len % STORAGE_DEFT_SZ) {
        cap = ((unsigned long long)((len) / STORAGE_DEFT_SZ) + 1) * STORAGE_DEFT_SZ;
      } else {
        cap = len;
      }
      cytnx_error_msg(((device == Device.cpu) == std::is_same_v<GpuAllocator<T>, Allocator>), "%s",
                      "[ERROR] Cannot switch between CPU and GPU in Init()."
                      " device:%d this->device:%d",
                      device, this->device);
      this->device = device;
      if constexpr (std::is_same_v<GpuAllocator<T>, Allocator>) {
        checkCudaErrors(cudaSetDevice(device));
      }
      storage_.reserve(cap);
      storage_.resize(len);
      if constexpr (!std::is_same_v<bool, T>) {
        Mem = reinterpret_cast<void *>(storage_.data());
      }
    };

    // TODO: Remove this unsafe initializer.
    void _Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device = -1,
                     const bool &iscap = false, const unsigned long long &cap_in = 0) override {
      cytnx_error_msg(cap_in < len_in, "%s", "[ERROR] _Init_by_ptr cannot have capacity < size.");
      cytnx_error_msg(len_in < 1, "%s", "[ERROR] _Init_by_ptr cannot have len_in < 1.");
      cytnx_error_msg(cap_in % STORAGE_DEFT_SZ != 0,
                      "[ERROR] _Init_by_ptr cannot have not %dx cap_in.", STORAGE_DEFT_SZ);
      cytnx_error_msg(this->device != device, "[ERROR] Cannot change device by the raw pointer.",
                      STORAGE_DEFT_SZ);
      cap = iscap ? cap_in : len_in;
      len = len_in;
      storage_.reserve(cap);
      storage_.resize(len);
      Mem = storage_.data();
      auto *source_ptr = reinterpret_cast<value_type *>(rawptr);
      if constexpr (std::is_same_v<GpuAllocator<T>, Allocator>) {
        checkCudaErrors(cudaDeviceSynchronize());
      }
      std::copy(source_ptr, source_ptr + len, storage_.begin());
      if constexpr (std::is_same_v<GpuAllocator<T>, Allocator>) {
        checkCudaErrors(cudaFree(rawptr));
      } else {
        free(rawptr);
      }
    };

    // will be removed
    boost::intrusive_ptr<Storage_base> _create_new_sametype() {
      return boost::intrusive_ptr<Storage_base>(new StorageImplementation(device));
    };
    // will be removed
    boost::intrusive_ptr<Storage_base> clone() override {
      if constexpr (std::is_same_v<GpuAllocator<T>, Allocator>) {
        checkCudaErrors(cudaSetDevice(device));
        checkCudaErrors(cudaDeviceSynchronize());
      }
      return boost::intrusive_ptr<Storage_base>(new StorageImplementation(*this));
    };

    boost::intrusive_ptr<Storage_base> Movemem_cpu(boost::intrusive_ptr<Storage_base> &in,
                                                   const std::vector<cytnx_uint64> &old_shape,
                                                   const std::vector<cytnx_uint64> &mapper,
                                                   const std::vector<cytnx_uint64> &invmapper,
                                                   const bool is_inplace) {
      auto old = boost::dynamic_pointer_cast<StorageImplementation>(in);
      std::vector<cytnx_uint64> offsets(old_shape.size());
      int current_offset = 1;
      for (auto idx = mapper.rbegin(); idx != mapper.rend(); ++idx) {
        offsets[*idx] = current_offset;
        current_offset *= old_shape[*idx];
      }
      int actual_size = current_offset;  // TODO: get from old.size()
      std::vector<int> current_indices(old_shape.size());
      if (!is_inplace) {
        auto *out = new StorageImplementation(actual_size, old->device, /* init_zero */ false);
        int niddle = old_shape.size() - 1;
        int position = 0;
        for (int i = 0; i < actual_size - 1; ++i) {
          out->storage_[position] = old->storage_[i];
          niddle = old_shape.size() - 1;
          while (current_indices[niddle] == old_shape[niddle] - 1) {
            position -= current_indices[niddle] * offsets[niddle];
            current_indices[niddle--] = 0;
          }
          current_indices[niddle] += 1;
          position += offsets[niddle];
        }
        out->storage_[actual_size - 1] = old->storage_[actual_size - 1];
        return boost::intrusive_ptr<Storage_base>(out);
      }
      std::vector<bool> seen(actual_size);
      std::vector<cytnx_uint64> original_offsets(old_shape.size());
      current_offset = 1;
      original_offsets[old_shape.size() - 1] = current_offset;
      for (size_t i = old_shape.size() - 1; i > 0; --i) {
        current_offset *= old_shape[i];
        original_offsets[i - 1] = current_offset;
      }

      for (int i = 0; i < seen.size(); ++i) {
        int position = i;
        while (!seen[i]) {
          seen[i] = true;
          int new_position = 0;
          for (size_t j = 0; j < old_shape.size(); ++j) {
            new_position += position / original_offsets[j] * offsets[j];
            position %= original_offsets[j];
          }
          std::swap(old->storage_[i], old->storage_[new_position]);
          position = new_position;
        }
      }
      return in;
    }

    void Move_memory_(const std::vector<cytnx_uint64> &old_shape,
                      const std::vector<cytnx_uint64> &mapper,
                      const std::vector<cytnx_uint64> &invmapper) override {
      boost::intrusive_ptr<Storage_base> tmp(this);
      if constexpr (!std::is_same_v<GpuAllocator<value_type>, allocator_type>) {  // on CPU
        Movemem_cpu(tmp, old_shape, mapper, invmapper, /* is_inplace */ true);
      } else {
        if constexpr (std::is_same_v<value_type, cytnx_complex128>) {
          utils_internal::cuMovemem_gpu_cd(tmp, old_shape, mapper, invmapper,
                                           /* is_inplace */ true);
        } else if (std::is_same_v<value_type, cytnx_complex64>) {
          utils_internal::cuMovemem_gpu_cf(tmp, old_shape, mapper, invmapper,
                                           /* is_inplace */ true);
        } else if (std::is_same_v<value_type, cytnx_double>) {
          utils_internal::cuMovemem_gpu_d(tmp, old_shape, mapper, invmapper,
                                          /* is_inplace */ true);
        } else if (std::is_same_v<value_type, cytnx_float>) {
          utils_internal::cuMovemem_gpu_f(tmp, old_shape, mapper, invmapper,
                                          /* is_inplace */ true);
        } else if (std::is_same_v<value_type, cytnx_int64>) {
          utils_internal::cuMovemem_gpu_i64(tmp, old_shape, mapper, invmapper,
                                            /* is_inplace */ true);
        } else if (std::is_same_v<value_type, cytnx_uint64>) {
          utils_internal::cuMovemem_gpu_u64(tmp, old_shape, mapper, invmapper,
                                            /* is_inplace */ true);
        } else if (std::is_same_v<value_type, cytnx_int32>) {
          utils_internal::cuMovemem_gpu_i32(tmp, old_shape, mapper, invmapper,
                                            /* is_inplace */ true);
        } else if (std::is_same_v<value_type, cytnx_uint32>) {
          utils_internal::cuMovemem_gpu_u32(tmp, old_shape, mapper, invmapper,
                                            /* is_inplace */ true);
        } else if (std::is_same_v<value_type, cytnx_int16>) {
          utils_internal::cuMovemem_gpu_i16(tmp, old_shape, mapper, invmapper,
                                            /* is_inplace */ true);
        } else if (std::is_same_v<value_type, cytnx_uint16>) {
          utils_internal::cuMovemem_gpu_u16(tmp, old_shape, mapper, invmapper,
                                            /* is_inplace */ true);
        } else if (std::is_same_v<value_type, char>) {  // for cytnx_bool
          utils_internal::cuMovemem_gpu_b(tmp, old_shape, mapper, invmapper,
                                          /* is_inplace */ true);
        } else {
          assert(false);
        }
      }
    };

    boost::intrusive_ptr<Storage_base> Move_memory(
      const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper,
      const std::vector<cytnx_uint64> &invmapper) override {
      boost::intrusive_ptr<Storage_base> tmp(this);
      if constexpr (!std::is_same_v<GpuAllocator<value_type>, allocator_type>) {  // on CPU
        return Movemem_cpu(tmp, old_shape, mapper, invmapper, /* is_inplace */ false);
      } else {
        if constexpr (std::is_same_v<value_type, cytnx_complex128>) {
          return utils_internal::cuMovemem_gpu_cd(tmp, old_shape, mapper, invmapper,
                                                  /* is_inplace */ false);
        } else if (std::is_same_v<value_type, cytnx_complex64>) {
          return utils_internal::cuMovemem_gpu_cf(tmp, old_shape, mapper, invmapper,
                                                  /* is_inplace */ false);
        } else if (std::is_same_v<value_type, cytnx_double>) {
          return utils_internal::cuMovemem_gpu_d(tmp, old_shape, mapper, invmapper,
                                                 /* is_inplace */ false);
        } else if (std::is_same_v<value_type, cytnx_float>) {
          return utils_internal::cuMovemem_gpu_f(tmp, old_shape, mapper, invmapper,
                                                 /* is_inplace */ false);
        } else if (std::is_same_v<value_type, cytnx_int64>) {
          return utils_internal::cuMovemem_gpu_i64(tmp, old_shape, mapper, invmapper,
                                                   /* is_inplace */ false);
        } else if (std::is_same_v<value_type, cytnx_uint64>) {
          return utils_internal::cuMovemem_gpu_u64(tmp, old_shape, mapper, invmapper,
                                                   /* is_inplace */ false);
        } else if (std::is_same_v<value_type, cytnx_int32>) {
          return utils_internal::cuMovemem_gpu_i32(tmp, old_shape, mapper, invmapper,
                                                   /* is_inplace */ false);
        } else if (std::is_same_v<value_type, cytnx_uint32>) {
          return utils_internal::cuMovemem_gpu_u32(tmp, old_shape, mapper, invmapper,
                                                   /* is_inplace */ false);
        } else if (std::is_same_v<value_type, cytnx_int16>) {
          return utils_internal::cuMovemem_gpu_i16(tmp, old_shape, mapper, invmapper,
                                                   /* is_inplace */ false);
        } else if (std::is_same_v<value_type, cytnx_uint16>) {
          return utils_internal::cuMovemem_gpu_u16(tmp, old_shape, mapper, invmapper,
                                                   /* is_inplace */ false);
        } else if (std::is_same_v<value_type, char>) {  // for cytnx_bool
          return utils_internal::cuMovemem_gpu_b(tmp, old_shape, mapper, invmapper,
                                                 /* is_inplace */ false);
        } else {
          assert(false);
        }
      }
    }

    // will be removed
    void to_(const int &device) override{};
    // will be removed
    boost::intrusive_ptr<Storage_base> to(const int &device) override {
  #ifndef UNI_GPU
      cytnx_error_msg(device != Device.cpu,
                      "[ERROR] Cannot transfer to GPU with the binary built without CUDA."
                      " device:%d",
                      device);
      return nullptr;
  #endif  // UNI_GPU
      bool to_cpu = device == Device.cpu;
      bool this_is_on_cpu = !std::is_same_v<GpuAllocator<T>, Allocator>;
      if (to_cpu == this_is_on_cpu) {  // also catches the case transferring between two GPUs
        return this;
      }
      boost::intrusive_ptr<Storage_base> dest;
      if (device == Device.cpu) {
        dest = new StorageImplementation<value_type>(device);
      } else {
        dest = new StorageImplementation<value_type, GpuAllocator<value_type>>(device);
      }
      dest->Init(storage_.size(), device, /* init_zero */ false);
      if constexpr (std::is_same_v<GpuAllocator<T>, Allocator>) {
        checkCudaErrors(cudaSetDevice(this->device));
        checkCudaErrors(cudaDeviceSynchronize());
      }
      std::memcpy(dest->data(), data(), storage_.size() * sizeof(value_type));
      return dest;
    };

    void PrintElem_byShape(std::ostream &os, const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &mapper = {}) {
      if constexpr (std::is_same_v<GpuAllocator<T>, Allocator>) {
        checkCudaErrors(cudaSetDevice(device));
        checkCudaErrors(cudaDeviceSynchronize());
      }
      os << "\nTotal elem: " << size() << '\n';
      os << "type  : " << Type.getname(dtype) << '\n';
      os << Device.getname(device) << '\n';
      os << "Shape :(";
      if (!shape.empty()) {
        os << shape[0];
        for (auto it = shape.begin() + 1; it != shape.end(); ++it) {
          os << ", " << *it;
        }
      }
      os << ")\n";
      if (storage_.empty()) {
        os << "[]" << std::endl;
        return;
      }

      os << "[";
      std::vector<cytnx_uint64> origin_to_now;
      if (mapper.empty()) {
        origin_to_now.reserve(shape.size());
        for (cytnx_uint64 i = 0; i < shape.size(); ++i) {
          origin_to_now.push_back(i);
        }
      } else {
        origin_to_now = mapper;
      }
      // The summation of all offsets[i] * current_indices[i] is the position of the number
      // represetning by current_indices.
      std::vector<int> offsets(shape.size());
      int current_offset = 1;
      offsets[origin_to_now[shape.size() - 1]] = current_offset;
      for (size_t i = shape.size(); --i;) {
        current_offset *= shape[origin_to_now[i]];
        offsets[origin_to_now[i - 1]] = current_offset;
      }
      std::vector<int> current_indices(shape.size());
      int position = 0;
      size_t niddle = 0;
      for (const_iterator it = storage_.begin(); it < storage_.end() - 1; ++it) {
        // output extra newline when finishing to output a matrix and a array
        if (it != storage_.begin()) {
          if (shape.size() - niddle > 2) {
            os << std::string(2, '\n');
          } else if (shape.size() - niddle == 2) {
            os << '\n';
          }
        }
        if (niddle < shape.size() - 1) {
          os << std::string(niddle + 1, ' ') << std::string(shape.size() - niddle - 1, '[');
        } else {
          os << " ";  // the space between numbers in the same row
        }
        os << storage_[position];
        niddle = shape.size() - 1;
        while (current_indices[niddle] == shape[niddle] - 1) {
          position -= current_indices[niddle] * offsets[niddle];
          current_indices[niddle--] = 0;
        }
        current_indices[niddle] += 1;
        position += offsets[niddle];
        if (niddle < shape.size() - 1) {
          os << std::string(shape.size() - niddle - 1, ']');
        }
      }
      if (shape.back() > 1) {
        os << ' ';
      }
      os << storage_.back() << std::string(shape.size(), ']') << std::endl;
    };
    void print_elems() override {
      if constexpr (std::is_same_v<GpuAllocator<T>, Allocator>) {
        checkCudaErrors(cudaSetDevice(device));
        checkCudaErrors(cudaDeviceSynchronize());
      }
      std::cout << "[ ";
      for (const auto &value : storage_) {
        std::cout << value << " ";
      }
      std::cout << " ]";
    };

    iterator begin() noexcept { return storage_.begin(); }

    const_iterator begin() const noexcept { return storage_.begin(); }

    // TODO: real() and imag() should be moved to Tensor
    // TODO: The interface should be rewritten to prevent runtime type checking.
    boost::intrusive_ptr<Storage_base> real() override {
      if constexpr (std::is_same_v<cytnx_complex128, T> || std::is_same_v<cytnx_complex64, T>) {
        auto *out = new StorageImplementation<
          typename T::value_type,
          typename std::allocator_traits<Allocator>::template rebind_alloc<typename T::value_type>>(
          storage_.size(), device);
        if constexpr (std::is_same_v<GpuAllocator<T>, Allocator>) {
          checkCudaErrors(cudaSetDevice(device));
          if constexpr (std::is_same_v<cytnx_complex128, T>) {
            utils_internal::cuComplexmem_gpu_cdtd(out->data(), storage_.data(), storage_.size(),
                                                  /* get_real */ true);
          } else {
            utils_internal::cuComplexmem_gpu_cftf(out->data(), storage_.data(), storage_.size(),
                                                  /* get_real */ true);
          }
        } else {
          std::transform(
            storage_.begin(), storage_.end(), out->begin(),
            [](const auto &c) -> auto{ return c.real(); });
        }

        return boost::intrusive_ptr<Storage_base>(out);
      } else {
        cytnx_error_msg(true, "[ERROR] Storage.real() can only be called from complex type.%s",
                        "\n");
      }
    };

    boost::intrusive_ptr<Storage_base> imag() override {
      if constexpr (std::is_same_v<cytnx_complex128, T> || std::is_same_v<cytnx_complex64, T>) {
        auto *out = new StorageImplementation<
          typename T::value_type,
          typename std::allocator_traits<Allocator>::template rebind_alloc<typename T::value_type>>(
          storage_.size(), device);
        if constexpr (std::is_same_v<GpuAllocator<T>, Allocator>) {
          checkCudaErrors(cudaSetDevice(device));
          if constexpr (std::is_same_v<cytnx_complex128, T>) {
            utils_internal::cuComplexmem_gpu_cdtd(out->data(), storage_.data(), storage_.size(),
                                                  /* get_real */ false);
          } else {
            utils_internal::cuComplexmem_gpu_cftf(out->data(), storage_.data(), storage_.size(),
                                                  /* get_real */ false);
          }
        } else {
          std::transform(
            storage_.begin(), storage_.end(), out->begin(),
            [](const auto &c) -> auto{ return c.imag(); });
        }
        return boost::intrusive_ptr<Storage_base>(out);
      } else {
        cytnx_error_msg(true, "[ERROR] Storage.imag() can only be called from complex type.%s",
                        "\n");
      }
    };

    void fill(const cytnx_complex128 &val) override { Fill(val); };
    void fill(const cytnx_complex64 &val) override { Fill(val); };
    void fill(const cytnx_double &val) override { Fill(val); };
    void fill(const cytnx_float &val) override { Fill(val); };
    void fill(const cytnx_int64 &val) override { Fill(val); };
    void fill(const cytnx_uint64 &val) override { Fill(val); };
    void fill(const cytnx_int32 &val) override { Fill(val); };
    void fill(const cytnx_uint32 &val) override { Fill(val); };
    void fill(const cytnx_int16 &val) override { Fill(val); };
    void fill(const cytnx_uint16 &val) override { Fill(val); };
    void fill(const cytnx_bool &val) override { Fill(val); };
    void set_zeros() override { Fill(0); };
    void resize(const cytnx_uint64 &newsize) override {
      if constexpr (std::is_same_v<GpuAllocator<T>, Allocator>) {
        if (newsize > storage_.capacity()) {
          checkCudaErrors(cudaSetDevice(device));
          checkCudaErrors(cudaDeviceSynchronize());
        }
      }
      storage_.resize(newsize);
      len = newsize;
      Mem = reinterpret_cast<void *>(storage_.data());
    };

    void append(const Scalar &val) override { Append(val); }
    void append(const cytnx_complex128 &val) override { Append(val); }
    void append(const cytnx_complex64 &val) override { Append(val); }
    void append(const cytnx_double &val) override { Append(val); }
    void append(const cytnx_float &val) override { Append(val); }
    void append(const cytnx_int64 &val) override { Append(val); }
    void append(const cytnx_uint64 &val) override { Append(val); }
    void append(const cytnx_int32 &val) override { Append(val); }
    void append(const cytnx_uint32 &val) override { Append(val); }
    void append(const cytnx_int16 &val) override { Append(val); }
    void append(const cytnx_uint16 &val) override { Append(val); }
    void append(const cytnx_bool &val) override { Append(val); }
    Scalar get_item(const cytnx_uint64 &in) const override {
      if constexpr (std::is_same_v<GpuAllocator<T>, Allocator>) {
        checkCudaErrors(cudaSetDevice(device));
        checkCudaErrors(cudaDeviceSynchronize());
      }
      return storage_[in];
    };

    void set_item(const cytnx_uint64 &idx, const Scalar &val) { SetItem(idx, val); }
    void set_item(const cytnx_uint64 &idx, const cytnx_complex128 &val) { SetItem(idx, val); }
    void set_item(const cytnx_uint64 &idx, const cytnx_complex64 &val) { SetItem(idx, val); }
    void set_item(const cytnx_uint64 &idx, const cytnx_double &val) { SetItem(idx, val); }
    void set_item(const cytnx_uint64 &idx, const cytnx_float &val) { SetItem(idx, val); }
    void set_item(const cytnx_uint64 &idx, const cytnx_int64 &val) { SetItem(idx, val); }
    void set_item(const cytnx_uint64 &idx, const cytnx_uint64 &val) { SetItem(idx, val); }
    void set_item(const cytnx_uint64 &idx, const cytnx_int32 &val) { SetItem(idx, val); }
    void set_item(const cytnx_uint64 &idx, const cytnx_uint32 &val) { SetItem(idx, val); }
    void set_item(const cytnx_uint64 &idx, const cytnx_int16 &val) { SetItem(idx, val); }
    void set_item(const cytnx_uint64 &idx, const cytnx_uint16 &val) { SetItem(idx, val); }
    void set_item(const cytnx_uint64 &idx, const cytnx_bool &val) { SetItem(idx, val); }

   private:
    template <typename U>
    void Fill(U value) {
      if constexpr (std::is_constructible_v<T, U>) {
        if constexpr (std::is_same_v<GpuAllocator<T>, Allocator>) {
          checkCudaErrors(cudaSetDevice(device));
          value_type converted_value{value};
          if constexpr (std::is_same_v<value_type, cytnx_complex128>) {
            utils_internal::cuFill_gpu_cd(Mem, (void *)(&converted_value), len);
          } else if (std::is_same_v<value_type, cytnx_complex64>) {
            utils_internal::cuFill_gpu_cf(Mem, (void *)(&converted_value), len);
          } else if (std::is_same_v<value_type, cytnx_double>) {
            utils_internal::cuFill_gpu_d(Mem, (void *)(&converted_value), len);
          } else if (std::is_same_v<value_type, cytnx_float>) {
            utils_internal::cuFill_gpu_f(Mem, (void *)(&converted_value), len);
          } else if (std::is_same_v<value_type, cytnx_int64>) {
            utils_internal::cuFill_gpu_i64(Mem, (void *)(&converted_value), len);
          } else if (std::is_same_v<value_type, cytnx_uint64>) {
            utils_internal::cuFill_gpu_u64(Mem, (void *)(&converted_value), len);
          } else if (std::is_same_v<value_type, cytnx_int32>) {
            utils_internal::cuFill_gpu_i32(Mem, (void *)(&converted_value), len);
          } else if (std::is_same_v<value_type, cytnx_uint32>) {
            utils_internal::cuFill_gpu_u32(Mem, (void *)(&converted_value), len);
          } else if (std::is_same_v<value_type, cytnx_int16>) {
            utils_internal::cuFill_gpu_i16(Mem, (void *)(&converted_value), len);
          } else if (std::is_same_v<value_type, cytnx_uint16>) {
            utils_internal::cuFill_gpu_u16(Mem, (void *)(&converted_value), len);
          } else {
            assert(false);
          }
          checkCudaErrors(cudaDeviceSynchronize());
        } else {
          std::fill(storage_.begin(), storage_.end(), static_cast<T>(value));
        }
      } else {
        cytnx_error_msg(true, "%s", "[ERROR] Failed to convert the value.");
      }
    }

    template <typename U>
    void Append(U value) {
      if constexpr (std::is_constructible_v<T, U>) {
        if constexpr (std::is_same_v<GpuAllocator<T>, Allocator>) {
          // `push_back` may cause reallocation on the CPU side, so we have to wait GPU finishing
          // their jobs.
          checkCudaErrors(cudaSetDevice(device));
          checkCudaErrors(cudaDeviceSynchronize());
        }
        storage_.push_back(static_cast<T>(value));
        ++len;
        Mem = reinterpret_cast<void *>(storage_.data());
      } else {
        cytnx_error_msg(true, "%s", "[ERROR] Failed to convert the value.");
      }
    }

    template <typename U>
    void SetItem(size_type idx, U value) {
      if constexpr (std::is_constructible_v<T, U>) {
        storage_[idx] = static_cast<T>(value);
      } else {
        cytnx_error_msg(true, "%s", "[ERROR] Failed to convert the value.");
      }
    }

    storage_type storage_;
  };

  template <>
  Scalar StorageImplementation<char>::get_item(const cytnx_uint64 &in) const;

  template <>
  Scalar StorageImplementation<char, GpuAllocator<char>>::get_item(const cytnx_uint64 &in) const;

  template <>
  template <typename U>
  void StorageImplementation<char>::Fill(U value) {
    if constexpr (std::is_constructible_v<cytnx_bool, U>) {
      std::fill(storage_.begin(), storage_.end(),
                static_cast<char>(static_cast<cytnx_bool>(value)));
    } else {
      cytnx_error_msg(true, "%s", "[ERROR] Failed to convert the value.");
    }
  }

  template <>
  template <typename U>
  void StorageImplementation<char, GpuAllocator<char>>::Fill(U value) {
    if constexpr (std::is_constructible_v<cytnx_bool, U>) {
      checkCudaErrors(cudaSetDevice(device));
      bool converted_value{value};
      utils_internal::cuFill_gpu_b(this->Mem, (void *)(&converted_value), this->len);
    } else {
      cytnx_error_msg(true, "%s", "[ERROR] Failed to convert the value.");
    }
  }

  template <>
  template <typename U>
  void StorageImplementation<char>::Append(U value) {
    if constexpr (std::is_constructible_v<cytnx_bool, U>) {
      storage_.push_back(static_cast<char>(static_cast<cytnx_bool>(value)));
      ++len;
      Mem = reinterpret_cast<void *>(storage_.data());
    } else {
      cytnx_error_msg(true, "%s", "[ERROR] Failed to convert the value.");
    }
  }

  template <>
  template <typename U>
  void StorageImplementation<char, GpuAllocator<char>>::Append(U value) {
    if constexpr (std::is_constructible_v<cytnx_bool, U>) {
      // `push_back` may cause reallocation on the CPU side, so we have to wait GPU finishing their
      // jobs.
      checkCudaErrors(cudaSetDevice(device));
      checkCudaErrors(cudaDeviceSynchronize());
      storage_.push_back(static_cast<char>(static_cast<cytnx_bool>(value)));
      ++len;
      Mem = reinterpret_cast<void *>(storage_.data());
    } else {
      cytnx_error_msg(true, "%s", "[ERROR] Failed to convert the value.");
    }
  }

  template <>
  template <typename U>
  void StorageImplementation<char>::SetItem(size_type idx, U value) {
    if constexpr (std::is_constructible_v<cytnx_bool, U>) {
      storage_[idx] = static_cast<char>(static_cast<cytnx_bool>(value));
    } else {
      cytnx_error_msg(true, "%s", "[ERROR] Failed to convert the value.");
    }
  }

  template <>
  template <typename U>
  void StorageImplementation<char, GpuAllocator<char>>::SetItem(size_type idx, U value) {
    if constexpr (std::is_constructible_v<cytnx_bool, U>) {
      storage_[idx] = static_cast<char>(static_cast<cytnx_bool>(value));
    } else {
      cytnx_error_msg(true, "%s", "[ERROR] Failed to convert the value.");
    }
  }

  using ComplexDoubleStorage = StorageImplementation<cytnx_complex128>;
  using ComplexFloatStorage = StorageImplementation<cytnx_complex64>;
  using DoubleStorage = StorageImplementation<cytnx_double>;
  using FloatStorage = StorageImplementation<cytnx_float>;
  using Int64Storage = StorageImplementation<cytnx_int64>;
  using Uint64Storage = StorageImplementation<cytnx_uint64>;
  using Int32Storage = StorageImplementation<cytnx_int32>;
  using Uint32Storage = StorageImplementation<cytnx_uint32>;
  using Int16Storage = StorageImplementation<cytnx_int16>;
  using Uint16Storage = StorageImplementation<cytnx_uint16>;
  // Current implementation depends on std::vector<T>::data(), which is deleted for T = bool.
  // We store data in char and specilize member functions if needed.
  using BoolStorage = StorageImplementation<char>;

  #ifdef UNI_GPU
  using ComplexDoubleGpuStorage =
    StorageImplementation<cytnx_complex128, GpuAllocator<cytnx_complex128>>;
  using ComplexFloatGpuStorage =
    StorageImplementation<cytnx_complex64, GpuAllocator<cytnx_complex64>>;
  using DoubleGpuStorage = StorageImplementation<cytnx_double, GpuAllocator<cytnx_double>>;
  using FloatGpuStorage = StorageImplementation<cytnx_float, GpuAllocator<cytnx_float>>;
  using Int64GpuStorage = StorageImplementation<cytnx_int64, GpuAllocator<cytnx_int64>>;
  using Uint64GpuStorage = StorageImplementation<cytnx_uint64, GpuAllocator<cytnx_uint64>>;
  using Int32GpuStorage = StorageImplementation<cytnx_int32, GpuAllocator<cytnx_int32>>;
  using Uint32GpuStorage = StorageImplementation<cytnx_uint32, GpuAllocator<cytnx_uint32>>;
  using Int16GpuStorage = StorageImplementation<cytnx_int16, GpuAllocator<cytnx_int16>>;
  using Uint16GpuStorage = StorageImplementation<cytnx_uint16, GpuAllocator<cytnx_uint16>>;
  // Current implementation depends on std::vector<T>::data(), which is deleted for T = bool.
  // We store data in char and specilize member functions if needed.
  using BoolGpuStorage = StorageImplementation<char, GpuAllocator<char>>;
  #endif  // UNI_GPU

  ///@cond
  typedef boost::intrusive_ptr<Storage_base> (*pStorage_init)(int device);
  inline boost::intrusive_ptr<Storage_base> SIInit_cd(int device) {
    return device == Device.cpu
             ? boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage(device))
             : boost::intrusive_ptr<Storage_base>(new ComplexDoubleGpuStorage(device));
  }
  inline boost::intrusive_ptr<Storage_base> SIInit_cf(int device) {
    return device == Device.cpu
             ? boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage(device))
             : boost::intrusive_ptr<Storage_base>(new ComplexFloatGpuStorage(device));
  }
  inline boost::intrusive_ptr<Storage_base> SIInit_d(int device) {
    return device == Device.cpu ? boost::intrusive_ptr<Storage_base>(new DoubleStorage(device))
                                : boost::intrusive_ptr<Storage_base>(new DoubleGpuStorage(device));
  }
  inline boost::intrusive_ptr<Storage_base> SIInit_f(int device) {
    return device == Device.cpu ? boost::intrusive_ptr<Storage_base>(new FloatStorage(device))
                                : boost::intrusive_ptr<Storage_base>(new FloatGpuStorage(device));
  }
  inline boost::intrusive_ptr<Storage_base> SIInit_u64(int device) {
    return device == Device.cpu ? boost::intrusive_ptr<Storage_base>(new Uint64Storage(device))
                                : boost::intrusive_ptr<Storage_base>(new Uint64GpuStorage(device));
  }
  inline boost::intrusive_ptr<Storage_base> SIInit_i64(int device) {
    return device == Device.cpu ? boost::intrusive_ptr<Storage_base>(new Int64Storage(device))
                                : boost::intrusive_ptr<Storage_base>(new Int64GpuStorage(device));
  }
  inline boost::intrusive_ptr<Storage_base> SIInit_u32(int device) {
    return device == Device.cpu ? boost::intrusive_ptr<Storage_base>(new Uint32Storage(device))
                                : boost::intrusive_ptr<Storage_base>(new Uint32GpuStorage(device));
  }
  inline boost::intrusive_ptr<Storage_base> SIInit_i32(int device) {
    return device == Device.cpu ? boost::intrusive_ptr<Storage_base>(new Int32Storage(device))
                                : boost::intrusive_ptr<Storage_base>(new Int32GpuStorage(device));
  }
  inline boost::intrusive_ptr<Storage_base> SIInit_u16(int device) {
    return device == Device.cpu ? boost::intrusive_ptr<Storage_base>(new Uint16Storage(device))
                                : boost::intrusive_ptr<Storage_base>(new Uint16GpuStorage(device));
  }
  inline boost::intrusive_ptr<Storage_base> SIInit_i16(int device) {
    return device == Device.cpu ? boost::intrusive_ptr<Storage_base>(new Int16Storage(device))
                                : boost::intrusive_ptr<Storage_base>(new Int16GpuStorage(device));
  }
  inline boost::intrusive_ptr<Storage_base> SIInit_b(int device) {
    return device == Device.cpu ? boost::intrusive_ptr<Storage_base>(new BoolStorage(device))
                                : boost::intrusive_ptr<Storage_base>(new BoolGpuStorage(device));
  }
  ///@endcond
  ///@cond
  class Storage_init_interface : public Type_class {
   public:
    // std::vector<pStorage_init> USIInit;
    inline static pStorage_init USIInit[N_Type];
    inline static bool inited = false;
    Storage_init_interface() {
      if (!inited) {
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
        inited = true;
      }
    }
  };
  extern Storage_init_interface __SII;
  ///@endcond;

  ///@brief an memeory storage with multi-type/multi-device support
  class Storage {
   private:
    // Interface:
    // Storage_init_interface __SII;

   public:
    ///@cond
    boost::intrusive_ptr<Storage_base> _impl;
    ///@endcond

    /**
    @brief initialize a Storage
    @param[in] size the number of elements for the Storage
    @param[in] dtype the dtype of the Storage instance. This can be any of type defined in
    cytnx::Type
    @param[in] device the device of the Storage instance. This can be cytnx::Device.cpu or
    cytnx::Device.cuda+<gpuid> (see cytnx::Device for more details)

    ## Example:
    ### c++ API:
    \include example/Storage/Init.cpp
    #### output>
    \verbinclude example/Storage/Init.cpp.out
    ### python API:
    \include example/Storage/Init.py
    #### output>
    \verbinclude example/Storage/Init.py.out
    */
    void Init(const unsigned long long &size, const unsigned int &dtype = Type.Double,
              int device = -1, const bool &init_zero = true) {
      cytnx_error_msg(dtype >= N_Type, "%s", "[ERROR] invalid argument: dtype");
      this->_impl = __SII.USIInit[dtype](device);
      this->_impl->Init(size, device, init_zero);
    }
    // void _Init_byptr(void *rawptr, const unsigned long long &len_in, const unsigned int &dtype =
    // Type.Double, const int &device = -1,
    //                              const bool &iscap = false, const unsigned long long &cap_in =
    //                              0){
    //   cytnx_error_msg(dtype >= N_Type, "%s", "[ERROR] invalid argument: dtype");
    //   this->_impl = __SII.USIInit[dtype]();
    //   this->_impl->_Init_byptr(rawptr, len_in, device, iscap, cap_in);
    // }

    /**
     * @brief The constructor of Storage class. It will call the function
     * @ref Init(const unsigned long long &size, const unsigned int &dtype,
     *           int device, const bool &init_zero) "Init"
     * to initialize the Storage instance.
     * @see Init(const unsigned long long &size, const unsigned int &dtype, int device, const bool
     * &init_zero)
     */
    Storage(const unsigned long long &size, const unsigned int &dtype = Type.Double,
            int device = -1, const bool &init_zero = true) {
      Init(size, dtype, device, init_zero);
    }
    // Storage(void *rawptr, const unsigned long long &len_in, const unsigned int &dtype =
    // Type.Double, const int &device = -1,
    //       const bool &iscap = false, const unsigned long long &cap_in = 0)
    //       : _impl(new Storage_base()){
    //   _Init_byptr(rawptr,len_in,dtype,device,iscap,cap_in);
    // }

    /**
     * @brief The default constructor of Storage class. It will create an empty Storage instance.
     */
    Storage() : _impl(new Storage_base()){};
    ///@cond
    Storage(boost::intrusive_ptr<Storage_base> in_impl) { this->_impl = in_impl; }
    Storage(const Storage &rhs) { this->_impl = rhs._impl; }

    template <class Tp>
    Storage(const std::vector<Tp> &rhs) {
      this->_from_vector(rhs, -1);
    }
    template <class Tp>
    Storage(const std::initializer_list<Tp> &rhs) {
      this->_from_vector(std::vector<Tp>(rhs), -1);
    }

    Storage &operator=(const Storage &rhs) {
      this->_impl = rhs._impl;
      return *this;
    }

    ///@endcond

    /// @cond
    void _Save(std::fstream &f) const;
    void _Load(std::fstream &f);
    void _Loadbinary(std::fstream &f, const unsigned int &dtype, const cytnx_uint64 &Nelem);
    void _Savebinary(std::fstream &f) const;

    /// @endcond

    /**
    @brief Save current Storage to file
    @param[in] fname file name
    @details
        Save the Storage to file with file path specify with input param \p fname with postfix
    ".cyst"
    @post The file extension will be ".cyst".
    */
    void Save(const std::string &fname) const;

    /**
     * @brief Save current Storage to file, same as \ref Save(const std::string &fname)
     */
    void Save(const char *fname) const;
    /**
     * @brief Save current Storage to a binary file, which only contains the raw data.
     * @see Fromfile(const std::string &fname, const unsigned int &dtype, const cytnx_int64 &count)
     */
    void Tofile(const std::string &fname) const;
    /// @see Tofile(const std::string &fname) const
    void Tofile(const char *fname) const;
    /// @see Tofile(const std::string &fname) const
    void Tofile(std::fstream &f) const;

    /**
    @brief Load current Storage from file
    @param[in] fname file name
    @details
        load the Storage from file with file path specify with input param 'fname'.
    @pre The file must be a Storage object, which is saved by the function
        Save(const std::string &fname) const.
    */
    static Storage Load(const std::string &fname);

    /**
     * @brief Load current Storage from file, same as \ref Load(const std::string &fname)
     */
    static Storage Load(const char *fname);
    /**
     * @brief Load the binary file, which only contains the raw data, to current Storage.
     * @details This function will load the binary file, which only contains the raw data,
     *     to current Storage with specified dtype and number of elements.
     * @param[in] fname file name
     * @param[in] dtype the data type of the binary file. See cytnx::Type.
     * @param[in] Nelem the number of elements you want to load from the binary file. If
     *   \p Nelem is -1, then it will load all the elements in the binary file.
     * @pre
     *  1. The @p dtype cannot be Type.Void.
     *  2. The @p dtype must be the same as the data type of the binary file.
     *  3. The @p Nelem cannot be 0.
     *  4. The @p Nelem cannot be larger than the number of elements in the binary file.
     *  5. The file name @p fname must be valid.
     *
     * @see Tofile(const std::string &fname) const
     */
    static Storage Fromfile(const std::string &fname, const unsigned int &dtype,
                            const cytnx_int64 &count = -1);

    /**
     * @see Fromfile(const std::string &fname, const unsigned int &dtype, const cytnx_int64 &count =
     * -1)
     */
    static Storage Fromfile(const char *fname, const unsigned int &dtype,
                            const cytnx_int64 &count = -1);

    /**
    @brief cast the type of current Storage
    @details
        1. if the new_type is the same as the dtype of current Storage, then return self;
           otherwise, return a new instance that has the same content as current Storage with
           dtype= \p new_type .
        2. the return Stoarge will be on the same device as the current Storage.
    @param[in] new_type the new type of the Storage instance. This can be any of type defined in
    cytnx::Type.
    @attention
        This function cannot convert the complex type to real one. Please use real() or
          imag() to get the real or imaginary part of a complex Storage.


    ## Example:
    ### c++ API:
    \include example/Storage/astype.cpp
    #### output>
    \verbinclude example/Storage/astype.cpp.out
    ### python API:
    \include example/Storage/astype.py
    #### output>
    \verbinclude example/Storage/astype.py.out
    */
    Storage astype(const unsigned int &new_type) const { return this->_impl->astype(new_type); }

    /**
    @brief the dtype-id of current Storage, see cytnx::Type for more details.
    @return [cytnx_uint64] the dtype-id.
    */
    const unsigned int &dtype() const { return this->_impl->dtype; }

    /**
    @brief the dtype (std::string) of current Storage, see cytnx::Type for more details.
    @return [std::string] dtype name
    */
    const std::string dtype_str() const {
      std::string out = this->_impl->dtype_str();
      return out;
    }
    /**
    @brief the device-id of current Storage, see cytnx::Device for more details.
    @return [cytnx_int64] the device-id.
    */
    const int &device() const { return this->_impl->device; }

    /**
    @brief the device (std::string) of current Storage, see cytnx::Device for more details.
    @return [std::string] device name
    */
    const std::string device_str() const {
      std::string out = this->_impl->device_str();
      return out;
    }

    /**
    @brief append a value
    @param[in] val the value to append. it can be any type defined in cytnx::Type
    @note cannot append a complex value into a real Storage.
    */
    template <class T>
    void append(const T &val) {
      return this->_impl->append(val);
    }

    ///@cond
    template <class T>  // this is c++ only
    T &at(const cytnx_uint64 &idx) const {
      return this->_impl->at<T>(idx);
    }

    const Scalar::Sproxy at(const cytnx_uint64 &idx) const {
      Scalar::Sproxy out(this->_impl, idx);
      return out;
    }
    Scalar::Sproxy at(const cytnx_uint64 &idx) {
      Scalar::Sproxy out(this->_impl, idx);
      return out;
    }

    template <class T>  // this is c++ only
    T &back() const {
      return this->_impl->back<T>();
    }

    const Scalar::Sproxy back() const {
      Scalar::Sproxy out(this->_impl, this->size() - 1);
      return out;
    }
    Scalar::Sproxy back() {
      Scalar::Sproxy out(this->_impl, this->size() - 1);
      return out;
    }

    template <class T>  // this is c++ only
    T *data() const {
      return this->_impl->data<T>();
    }

    void *data() const { return this->_impl->data(); }
    ///@endcond

    /**
    @brief resize the current Storage.
    @param[in] newsize.
    */
    void resize(const cytnx_uint64 &newsize) { this->_impl->resize(newsize); }

    /**
    @brief Move the current Storage to different deivce.
    @param[in] device the device-id. It can be any device defined in cytnx::Device.
    @see Storage::to()
    */
    void to_(const int &device) { this->_impl->to_(device); }

    /**
    @brief move a new Storage with same content as current Storage on different deivce.
    @param[in] device the device-id. It can be any device defined in cytnx::Device
    @note
        if the \p device is the same as the current Storage's device, return self.
    @see Storage::to_()
    */
    Storage to(const int &device) { return Storage(this->_impl->to(device)); }

    /**
    @brief return a deep copy of the current storage.
    @return
        [Storage]

    ## Example:
    ### c++ API:
    \include example/Storage/clone.cpp
    #### output>
    \verbinclude example/Storage/clone.cpp.out
    ### python API:
    \include example/Storage/clone.py
    #### output>
    \verbinclude example/Storage/clone.py.out
    */
    Storage clone() const { return Storage(this->_impl->clone()); }

    /**
    @brief the size ( no. of elements ) in the Storage
    @return [cytnx_uint64]

    */
    const unsigned long long &size() const { return this->_impl->len; }

    /**
    @brief the capacity in the Storage.
    @details the capacity is the actual allocated memory in the Storage. The behavior of
      capacity is similar to std::vector::capacity() in c++.
    @return [cytnx_uint64]

    */
    const unsigned long long &capacity() const { return this->_impl->cap; }

    /**
    @brief print the info of the Storage, including the device, dtype and size.

    */
    void print_info() const { this->_impl->print_info(); }
    /// @cond
    // this is a redundant function
    void print() const { this->_impl->print(); }
    /// @endcond

    /**
    @brief set all the elements to zero.
    @note  although it is also possible to use Storage.fill(0) to set all the elements to zero,
           using set_zeros will have significant faster performance.

    */
    void set_zeros() { this->_impl->set_zeros(); }

    /**
    @brief compare two Storage
    @details This function will compare the content between two Storage objects. It will compare the
        "value" of each element. Even the two Storage are different objects (different instance), if
        they have the same values, this function will return true.
    @param[in] Storage another Storage to compare to
    @note the == operator will compare the content between two storages. use cytnx::is() for
    checking two variables share the same instance.

    ## Example:
    ### c++ API:
    \include example/Storage/eq.cpp
    #### output>
    \verbinclude example/Storage/eq.cpp.out
    ### python API:
    \include example/Storage/eq.py
    #### output>
    \verbinclude example/Storage/eq.py.out
    */
    bool operator==(const Storage &rhs);

    /**
     * @brief The not-equal operator for Storage.
     */
    bool operator!=(const Storage &rhs);

    /**
    @brief set all the elements to the assigned value val
    @param[in] val the value to set on all the elements. it can be any type defined in cytnx::Type
    @note cannot assign a complex value into a real Storage.
    */
    template <class T>
    void fill(const T &val) {
      this->_impl->fill(val);
    }

    /**
    @brief renew/create a Storage using c++ vector.
    @param[in] vin the C++ vector with supported types.
    @note This function is C++ only
    */
    template <class T>
    static Storage from_vector(const std::vector<T> &vin, const int device = -1) {
      Storage out;
      out._from_vector(vin, device);
      return out;
    }

    /*
        @brief convert a Storage to C++ vector.

        [Note]
            This function is C++ only
    */
    /*
    template <class T>
    std::vector<T> vector() {
      T tmp;
      cytnx_error_msg(Type.cy_typeid(tmp) != this->dtype(),
                      "[ERROR] the dtype of current Storage does not match assigned vector type.%s",
                      "\n");

      std::vector<T> out(this->size());
      Storage S;
      if (this->device() != Device.cpu) {
        S = this->to(Device.cpu);
        memcpy(&out[0], S.data(), sizeof(T) * this->size());
      } else {
        memcpy(&out[0], this->data(), sizeof(T) * this->size());
      }

      return out;
    }
    */

    /**
    @brief renew/create a c++ std::vector using current Storage.
    @note This function is C++ only
    */
    template <class T>
    std::vector<T> vector();

    /// @cond

    template <class T>
    void _from_vector(const std::vector<T> &vin, const int device = -1) {
      // auto dispatch:
      // check:
      cytnx_error_msg(1, "[FATAL] ERROR unsupport type%s", "\n");
      // this->_impl->Init(vin.size(),device);
      // memcpy(this->_impl->Mem,&vin[0],sizeof(T)*vin.size());
    }

    void _from_vector(const std::vector<cytnx_complex128> &vin, const int device = -1) {
      this->_impl = __SII.USIInit[Type.ComplexDouble](device);
      this->_impl->Init(vin.size(), device);
      memcpy(this->_impl->Mem, &vin[0], sizeof(cytnx_complex128) * vin.size());
    }
    void _from_vector(const std::vector<cytnx_complex64> &vin, const int device = -1) {
      this->_impl = __SII.USIInit[Type.ComplexFloat](device);
      this->_impl->Init(vin.size(), device);
      memcpy(this->_impl->Mem, &vin[0], sizeof(cytnx_complex64) * vin.size());
    }
    void _from_vector(const std::vector<cytnx_double> &vin, const int device = -1) {
      this->_impl = __SII.USIInit[Type.Double](device);
      this->_impl->Init(vin.size(), device);
      memcpy(this->_impl->Mem, &vin[0], sizeof(cytnx_double) * vin.size());
    }
    void _from_vector(const std::vector<cytnx_float> &vin, const int device = -1) {
      this->_impl = __SII.USIInit[Type.Float](device);
      this->_impl->Init(vin.size(), device);
      memcpy(this->_impl->Mem, &vin[0], sizeof(cytnx_float) * vin.size());
    }
    void _from_vector(const std::vector<cytnx_uint64> &vin, const int device = -1) {
      this->_impl = __SII.USIInit[Type.Uint64](device);
      this->_impl->Init(vin.size(), device);
      memcpy(this->_impl->Mem, &vin[0], sizeof(cytnx_uint64) * vin.size());
    }
    void _from_vector(const std::vector<cytnx_int64> &vin, const int device = -1) {
      this->_impl = __SII.USIInit[Type.Int64](device);
      this->_impl->Init(vin.size(), device);
      memcpy(this->_impl->Mem, &vin[0], sizeof(cytnx_int64) * vin.size());
    }
    void _from_vector(const std::vector<cytnx_uint32> &vin, const int device = -1) {
      this->_impl = __SII.USIInit[Type.Uint32](device);
      this->_impl->Init(vin.size(), device);
      memcpy(this->_impl->Mem, &vin[0], sizeof(cytnx_uint32) * vin.size());
    }
    void _from_vector(const std::vector<cytnx_int32> &vin, const int device = -1) {
      this->_impl = __SII.USIInit[Type.Int32](device);
      this->_impl->Init(vin.size(), device);
      memcpy(this->_impl->Mem, &vin[0], sizeof(cytnx_int32) * vin.size());
    }
    void _from_vector(const std::vector<cytnx_uint16> &vin, const int device = -1) {
      this->_impl = __SII.USIInit[Type.Uint16](device);
      this->_impl->Init(vin.size(), device);
      memcpy(this->_impl->Mem, &vin[0], sizeof(cytnx_uint16) * vin.size());
    }
    void _from_vector(const std::vector<cytnx_int16> &vin, const int device = -1) {
      this->_impl = __SII.USIInit[Type.Int16](device);
      this->_impl->Init(vin.size(), device);
      memcpy(this->_impl->Mem, &vin[0], sizeof(cytnx_int16) * vin.size());
    }
    void _from_vector(const std::vector<cytnx_bool> &vin, const int device = -1) {
      this->_impl = __SII.USIInit[Type.Bool](device);
      this->_impl->Init(vin.size(), device);
      this->_impl->_cpy_bool(this->_impl->Mem, vin);
      // memcpy(this->_impl->Mem,vin.data(),sizeof(cytnx_bool)*vin.size());
    }
    /// @endcond

    /**
    @brief Get the real part form a Complex type Storage
    @note Cannot be called from a real type Storage.

    ## Example:
    ### c++ API:
    \include example/Storage/real.cpp
    #### output>
    \verbinclude example/Storage/real.cpp.out
    ### python API:
    \include example/Storage/real.py
    #### output>
    \verbinclude example/Storage/real.py.out
    */
    Storage real() const { return Storage(this->_impl->real()); };

    /**
    @brief Get the imaginary part form a Complex type Storage
    @note Cannot be called from a real type Storage.

    ## Example:
    ### c++ API:
    \include example/Storage/imag.cpp
    #### output>
    \verbinclude example/Storage/imag.cpp.out
    ### python API:
    \include example/Storage/imag.py
    #### output>
    \verbinclude example/Storage/imag.py.out
    */
    Storage imag() const { return Storage(this->_impl->imag()); };

    /**
     * @brief Get the element at the given index.
     * @param[in] idx The index of the element.
     * @return The element at the given index.
     */
    Scalar get_item(const cytnx_uint64 &idx) const { return this->_impl->get_item(idx); };

    /**
     * @brief Set the element at the given index.
     * @param[in] idx The index of the element.
     * @param[in] elem The element to be set.
     */
    template <class T>
    void set_item(const cytnx_uint64 &idx, const T &elem) {
      this->_impl->set_item(idx, elem);
    };

    /**
     * @brief The access operator for the Storage.
     * @param[in] idx The index of the element.
     */
    Scalar::Sproxy operator()(const cytnx_uint64 &idx);

    // /**
    //  * @brief Check if two Storage are equivalent in element.
    //  * @param[in] rhs The Storage to compare.
    //  * @note Two Storage are equivalent if they have the same type, and elements.
    //  * @return If two Storage are equivalent.
    //  */
    // bool approx_eq(const Storage &rhs, const cytnx_double tol = 0) const {
    //   return this->_impl->approx_eq(rhs._impl, tol);
    // };
  };

  ///@cond
  std::ostream &operator<<(std::ostream &os, const Storage &in);
  ///@endcond

}  // namespace cytnx

#endif  // BACKEND_TORCH
#endif  // BACKEND_STORAGE_H_
