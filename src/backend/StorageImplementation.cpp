#include "backend/Storage.hpp"

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <vector>

#include "boost/smart_ptr/intrusive_ptr.hpp"

#include "backend/Scalar.hpp"
#include "backend/utils_internal_cpu/Alloc_cpu.hpp"
#include "backend/utils_internal_cpu/Complexmem_cpu.hpp"
#include "backend/utils_internal_cpu/Fill_cpu.hpp"
#include "backend/utils_internal_cpu/Movemem_cpu.hpp"
#include "backend/utils_internal_cpu/SetZeros_cpu.hpp"
#include "backend/utils_internal_gpu/cuAlloc_gpu.hpp"
#include "backend/utils_internal_gpu/cuComplexmem_gpu.hpp"
#include "backend/utils_internal_gpu/cuFill_gpu.hpp"
#include "backend/utils_internal_gpu/cuMovemem_gpu.hpp"
#include "backend/utils_internal_gpu/cuSetZeros_gpu.hpp"
#include "cytnx_error.hpp"
#include "Device.hpp"
#include "Type.hpp"

#ifdef UNI_GPU
  #include "cuda_runtime_api.h"
#endif

using namespace std;
namespace cytnx {
  template <typename DType>
  void PrintValueAndSpace(std::ostream &os, const DType &value) {
    if constexpr (is_complex_v<DType>) {
      std::streamsize original_percision = os.precision(5);
      std::ios::fmtflags original_flags = os.flags(std::ios::scientific);
      os << value.real() << std::showpos << value.imag() << "j ";
      os.precision(original_percision);
      os.flags(original_flags);
    } else if constexpr (std::is_floating_point_v<DType>) {
      std::streamsize original_percision = os.precision(5);
      std::ios::fmtflags original_flags = os.flags(std::ios::scientific);
      os << value << " ";
      os.precision(original_percision);
      os.flags(original_flags);
    } else if constexpr (std::is_same_v<DType, cytnx_uint64>) {
      os << std::setw(19) << value << " ";
    } else if constexpr (std::is_same_v<DType, cytnx_int64>) {
      os << std::setw(19) << std::showpos << value << std::noshowpos << " ";
    } else if constexpr (std::is_same_v<DType, cytnx_uint32>) {
      os << std::setw(10) << value << " ";
    } else if constexpr (std::is_same_v<DType, cytnx_int32>) {
      os << std::setw(10) << std::showpos << value << std::noshowpos << " ";
    } else if constexpr (std::is_same_v<DType, cytnx_uint16>) {
      os << std::setw(5) << value << " ";
    } else if constexpr (std::is_same_v<DType, cytnx_int16>) {
      os << std::setw(5) << std::showpos << value << std::noshowpos << " ";
    } else if constexpr (std::is_same_v<DType, cytnx_bool>) {
      os << (value ? "True  " : "False ");
    }
  }

  template <typename DType>
  void StorageImplementation<DType>::Init(const unsigned long long &len_in, const int &device,
                                          const bool &init_zero) {
    this->len = len_in;

    // check:
    // cytnx_error_msg(len_in < 1, "%s", "[ERROR] cannot init a Storage with zero element");
    this->dtype = Type.cy_typeid(DType());

    if (this->len % STORAGE_DEFT_SZ) {
      this->cap = ((unsigned long long)((this->len) / STORAGE_DEFT_SZ) + 1) * STORAGE_DEFT_SZ;
    } else {
      this->cap = this->len;
    }

    if (device == Device.cpu) {
      if (init_zero)
        this->Mem = utils_internal::Calloc_cpu(this->cap, sizeof(DType));
      else
        this->Mem = utils_internal::Malloc_cpu(this->cap * sizeof(DType));
    } else {
#ifdef UNI_GPU
      cytnx_error_msg(device >= Device.Ngpus, "%s", "[ERROR] invalid device.");
      checkCudaErrors(cudaSetDevice(device));
      this->Mem = utils_internal::cuCalloc_gpu(this->cap, sizeof(DType));

#else
      cytnx_error_msg(1, "%s", "[ERROR] cannot init a Storage on gpu without CUDA support.");
#endif
    }
    this->device = device;
  }

  template <typename DType>
  void StorageImplementation<DType>::_Init_byptr(void *rawptr, const unsigned long long &len_in,
                                                 const int &device, const bool &iscap,
                                                 const unsigned long long &cap_in) {
    //[note], this is an internal function, the device should match the device_id that allocate the
    // pointer if the pointer is on GPU device.

    this->Mem = rawptr;
    this->len = len_in;
    if (iscap) {
      this->cap = cap_in;
    } else {
      this->cap = len_in;
    }

    cytnx_error_msg(this->cap % STORAGE_DEFT_SZ != 0,
                    "[ERROR] _Init_by_ptr cannot have not %dx cap_in.", STORAGE_DEFT_SZ);

#ifdef UNI_DEBUG
    cytnx_error_msg(len_in < 1, "%s", "[ERROR] _Init_by_ptr cannot have len_in < 1.");
    cytnx_error_msg(this->cap < this->len, "%s",
                    "[ERROR] _Init_by_ptr cannot have capacity < size.");
#endif
    this->dtype = Type.cy_typeid(DType());
    this->device = device;
  }

  template <typename DType>
  boost::intrusive_ptr<Storage_base> StorageImplementation<DType>::_create_new_sametype() {
    boost::intrusive_ptr<Storage_base> out(new StorageImplementation());
    return out;
  }

  template <typename DType>
  boost::intrusive_ptr<Storage_base> StorageImplementation<DType>::clone() {
    boost::intrusive_ptr<Storage_base> out(new StorageImplementation());
    out->Init(this->len, this->device);
    if (this->device == Device.cpu) {
      memcpy(out->Mem, this->Mem, sizeof(DType) * this->len);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      checkCudaErrors(
        cudaMemcpy(out->Mem, this->Mem, sizeof(DType) * this->len, cudaMemcpyDeviceToDevice));
#else
      cytnx_error_msg(1, "%s", "[ERROR] cannot clone a Storage on gpu without CUDA support.");
#endif
    }
    return out;
  }

  template <typename DType>
  void StorageImplementation<DType>::Move_memory_(const std::vector<cytnx_uint64> &old_shape,
                                                  const std::vector<cytnx_uint64> &mapper,
                                                  const std::vector<cytnx_uint64> &invmapper) {
    boost::intrusive_ptr<Storage_base> tmp(this);
    if (this->device == Device.cpu) {
      utils_internal::MoveMemoryCpu<DType>(tmp, old_shape, mapper, invmapper, 1);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      utils_internal::MoveMemoryGpu<DType>(tmp, old_shape, mapper, invmapper, 1);
#else
      cytnx_error_msg(1, "%s", "[ERROR][Internal] try to call GPU section without CUDA support");
#endif
    }
  }

  template <typename DType>
  boost::intrusive_ptr<Storage_base> StorageImplementation<DType>::Move_memory(
    const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper,
    const std::vector<cytnx_uint64> &invmapper) {
    boost::intrusive_ptr<Storage_base> tmp(this);
    if (this->device == Device.cpu) {
      return utils_internal::MoveMemoryCpu<DType>(tmp, old_shape, mapper, invmapper, 0);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      return utils_internal::MoveMemoryGpu<DType>(tmp, old_shape, mapper, invmapper, 0);
#else
      cytnx_error_msg(1, "%s", "[ERROR][Internal] try to call GPU section without CUDA support");
      return nullptr;
#endif
    }
  }

  template <typename DType>
  void StorageImplementation<DType>::to_(const int &device) {
    if (this->device != device) {
      if (this->device == Device.cpu) {
// here, cpu->gpu with gid=device
#ifdef UNI_GPU
        cytnx_error_msg(device >= Device.Ngpus, "%s", "[ERROR] invalid device.");
        cudaSetDevice(device);
        void *dtmp = utils_internal::cuMalloc_gpu(sizeof(DType) * this->cap);
        checkCudaErrors(
          cudaMemcpy(dtmp, this->Mem, sizeof(DType) * this->len, cudaMemcpyHostToDevice));
        free(this->Mem);
        this->Mem = dtmp;
        this->device = device;
#else
        cytnx_error_msg(1, "%s", "[ERROR] try to move from cpu(Host) to gpu without CUDA support.");
#endif
      } else {
#ifdef UNI_GPU
        if (device == Device.cpu) {
          // here, gpu->cpu
          cudaSetDevice(this->device);
          void *htmp = malloc(sizeof(DType) * this->cap);
          checkCudaErrors(
            cudaMemcpy(htmp, this->Mem, sizeof(DType) * this->len, cudaMemcpyDeviceToHost));
          cudaFree(this->Mem);
          this->Mem = htmp;
          this->device = device;
        } else {
          // here, gpu->gpu
          cytnx_error_msg(device >= Device.Ngpus, "%s", "[ERROR] invalid device.");
          cudaSetDevice(device);
          void *dtmp = utils_internal::cuMalloc_gpu(sizeof(DType) * this->cap);
          checkCudaErrors(
            cudaMemcpyPeer(dtmp, device, this->Mem, this->device, sizeof(DType) * this->len));
          cudaFree(this->Mem);
          this->Mem = dtmp;
          this->device = device;
        }
#else
        cytnx_error_msg(
          1, "%s",
          "[ERROR][Internal] Storage.to_. the Storage is as GPU but without CUDA support.");
#endif
      }
    }
  }
  template <typename DType>
  boost::intrusive_ptr<Storage_base> StorageImplementation<DType>::to(const int &device) {
    // Here, we follow pytorch scheme. if the device is the same as this->device, then return this
    // (python self) otherwise, return a clone on different device.
    if (this->device == device) {
      return this;
    } else {
      if (this->device == Device.cpu) {
// here, cpu->gpu with gid=device
#ifdef UNI_GPU
        cytnx_error_msg(device >= Device.Ngpus, "%s", "[ERROR] invalid device.");
        cudaSetDevice(device);
        void *dtmp = utils_internal::cuMalloc_gpu(sizeof(DType) * this->cap);
        checkCudaErrors(
          cudaMemcpy(dtmp, this->Mem, sizeof(DType) * this->len, cudaMemcpyHostToDevice));
        boost::intrusive_ptr<Storage_base> out(new StorageImplementation());
        out->_Init_byptr(dtmp, this->len, device, true, this->cap);
        return out;
#else
        cytnx_error_msg(1, "%s", "[ERROR] try to move from cpu(Host) to gpu without CUDA support.");
        return nullptr;
#endif
      } else {
#ifdef UNI_GPU
        if (device == Device.cpu) {
          // here, gpu->cpu
          cudaSetDevice(this->device);
          void *htmp = malloc(sizeof(DType) * this->cap);
          checkCudaErrors(
            cudaMemcpy(htmp, this->Mem, sizeof(DType) * this->len, cudaMemcpyDeviceToHost));
          boost::intrusive_ptr<Storage_base> out(new StorageImplementation());
          out->_Init_byptr(htmp, this->len, device, true, this->cap);
          return out;
        } else {
          // here, gpu->gpu
          cytnx_error_msg(device >= Device.Ngpus, "%s", "[ERROR] invalid device.");
          cudaSetDevice(device);
          void *dtmp = utils_internal::cuMalloc_gpu(sizeof(DType) * this->cap);
          checkCudaErrors(
            cudaMemcpyPeer(dtmp, device, this->Mem, this->device, sizeof(DType) * this->len));
          boost::intrusive_ptr<Storage_base> out(new StorageImplementation());
          out->_Init_byptr(dtmp, this->len, device, true, this->cap);
          return out;
        }
#else
        cytnx_error_msg(
          1, "%s",
          "[ERROR][Internal] Storage.to_. the Storage is as GPU but without CUDA support.");
        return nullptr;
#endif
      }
    }
  }

  template <typename DType>
  void StorageImplementation<DType>::PrintElem_byShape(std::ostream &os,
                                                       const std::vector<cytnx_uint64> &shape,
                                                       const std::vector<cytnx_uint64> &mapper) {
    char *buffer = (char *)malloc(sizeof(char) * 256);
    // checking:
    cytnx_uint64 Ne = 1;
    for (cytnx_uint64 i = 0; i < shape.size(); i++) {
      Ne *= shape[i];
    }
    if (Ne != this->len) {
      cytnx_error_msg(1, "%s",
                      "PrintElem_byShape, the number of shape not match with the No. of elements.");
    }

    if (len == 0) {
      os << "[ ";
      os << "\nThe Storage has not been allocated or linked.\n";
      os << "]\n";
    } else {
      os << std::endl << "Total elem: " << this->len << "\n";

      os << "type  : " << Type.getname(this->dtype) << std::endl;

      int atDevice = this->device;
      os << Device.getname(this->device) << std::endl;

      sprintf(buffer, "%s", "Shape :");
      os << std::string(buffer);
      sprintf(buffer, " (%llu", shape[0]);
      os << std::string(buffer);
      for (cytnx_int32 i = 1; i < shape.size(); i++) {
        sprintf(buffer, ",%llu", shape[i]);
        os << std::string(buffer);
      }
      os << ")" << std::endl;

      // temporary move to cpu for printing.
      if (this->device != Device.cpu) {
        this->to_(Device.cpu);
      }

      std::vector<cytnx_uint64> stk(shape.size(), 0), stk2;

      cytnx_uint64 s;
      DType *elem_ptr_ = reinterpret_cast<DType *>(this->Mem);

      if (mapper.empty()) {
        cytnx_uint64 cnt = 0;
        while (true) {
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            if (i < shape.size() - stk.size()) {
              sprintf(buffer, "%s", " ");
              os << std::string(buffer);
            } else {
              stk2.push_back(0);
              sprintf(buffer, "%s", "[");
              os << std::string(buffer);
              stk.pop_back();
            }
          }
          for (cytnx_uint64 i = 0; i < shape.back(); i++) {
            stk2.back() = i;
            PrintValueAndSpace(os, elem_ptr_[cnt]);
            cnt++;
          }

          s = 0;
          while (true) {
            if (stk2.empty()) {
              break;
            }
            if (stk2.back() == *(&shape.back() - s) - 1) {
              stk.push_back(*(&shape.back() - s));
              s++;
              stk2.pop_back();
              sprintf(buffer, "%s", "]");
              os << std::string(buffer);
            } else {
              stk2.back() += 1;
              break;
            }
          }
          os << "\n";

          if (stk2.empty()) break;
        }
        os << std::endl;

      } else {
        /// This is for non-contiguous Tensor printing;
        // cytnx_error_msg(1,"%s","print for a non-contiguous Storage is under developing");
        // cytnx_uint64 cnt=0;
        std::vector<cytnx_uint64> c_offj(shape.size());
        std::vector<cytnx_uint64> c_shape(shape.size());

        cytnx_uint64 accu = 1;
        cytnx_uint64 RealMemPos;
        for (cytnx_uint32 i = 0; i < shape.size(); i++) {
          c_shape[i] = shape[mapper[i]];
        }
        for (cytnx_int64 i = c_shape.size() - 1; i >= 0; i--) {
          c_offj[i] = accu;
          accu *= c_shape[i];
        }

        while (true) {
          for (cytnx_int32 i = 0; i < shape.size(); i++) {
            if (i < shape.size() - stk.size()) {
              sprintf(buffer, "%s", " ");
              os << std::string(buffer);
            } else {
              stk2.push_back(0);
              sprintf(buffer, "%s", "[");
              os << std::string(buffer);
              stk.pop_back();
            }
          }
          for (cytnx_uint64 i = 0; i < shape.back(); i++) {
            stk2.back() = i;

            /// Calculate the Memory reflection:
            RealMemPos = 0;
            for (cytnx_uint64 n = 0; n < shape.size(); n++) {
              RealMemPos += c_offj[n] * stk2[mapper[n]];  // mapback + backmap = normal-map
            }
            PrintValueAndSpace(os, elem_ptr_[RealMemPos]);
            // cnt++;
          }

          s = 0;
          while (1) {
            if (stk2.empty()) {
              break;
            }
            if (stk2.back() == *(&shape.back() - s) - 1) {
              stk.push_back(*(&shape.back() - s));
              s++;
              stk2.pop_back();
              sprintf(buffer, "%s", "]");
              os << std::string(buffer);
            } else {
              stk2.back() += 1;
              break;
            }
          }
          os << "\n";

          if (stk2.empty()) break;
        }
        os << std::endl;

      }  // check if need mapping

      if (atDevice != Device.cpu) {
        this->to_(atDevice);
      }

    }  // len==0
    free(buffer);
  }

  template <typename DType>
  void StorageImplementation<DType>::print_elems() {
    DType *elem_ptr_ = reinterpret_cast<DType *>(this->Mem);
    cout << "[ ";
    for (unsigned long long cnt = 0; cnt < this->len; cnt++) {
      PrintValueAndSpace(std::cout, elem_ptr_[cnt]);
    }
    std::cout << "]" << std::endl;
  }

  template <typename DType>
  void StorageImplementation<DType>::fill(const cytnx_complex128 &val) {
    Fill(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::fill(const cytnx_complex64 &val) {
    Fill(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::fill(const cytnx_double &val) {
    Fill(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::fill(const cytnx_float &val) {
    Fill(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::fill(const cytnx_uint64 &val) {
    Fill(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::fill(const cytnx_int64 &val) {
    Fill(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::fill(const cytnx_uint32 &val) {
    Fill(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::fill(const cytnx_int32 &val) {
    Fill(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::fill(const cytnx_uint16 &val) {
    Fill(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::fill(const cytnx_int16 &val) {
    Fill(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::fill(const cytnx_bool &val) {
    Fill(val);
  }

  template <typename DType>
  void StorageImplementation<DType>::set_zeros() {
    if (this->device == Device.cpu) {
      utils_internal::SetZeros(this->Mem, sizeof(DType) * this->len);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      utils_internal::cuSetZeros(this->Mem, sizeof(DType) * this->len);
#else
      cytnx_error_msg(1, "[ERROR][set_zeros] fatal, the storage is on gpu without CUDA support.%s",
                      "\n");
#endif
    }
  }

  template <typename DType>
  void StorageImplementation<DType>::resize(const cytnx_uint64 &newsize) {
    // cytnx_error_msg(newsize < 1,"[ERROR]resize should have size > 0%s","\n");

    if (newsize > this->cap) {
      if (newsize % STORAGE_DEFT_SZ) {
        this->cap = ((unsigned long long)((newsize) / STORAGE_DEFT_SZ) + 1) * STORAGE_DEFT_SZ;
      } else {
        this->cap = newsize;
      }
      if (this->device == Device.cpu) {
        void *htmp = calloc(this->cap, sizeof(DType));
        memcpy(htmp, this->Mem, sizeof(DType) * this->len);
        free(this->Mem);
        this->Mem = htmp;
      } else {
#ifdef UNI_GPU
        cytnx_error_msg(device >= Device.Ngpus, "%s", "[ERROR] invalid device.");
        cudaSetDevice(device);
        void *dtmp = utils_internal::cuCalloc_gpu(this->cap, sizeof(DType));
        checkCudaErrors(
          cudaMemcpyPeer(dtmp, device, this->Mem, this->device, sizeof(DType) * this->len));
        cudaFree(this->Mem);
        this->Mem = dtmp;
#else
        cytnx_error_msg(
          1, "%s",
          "[ERROR][Internal] Storage.resize. the Storage is as GPU but without CUDA support.");
#endif
      }
    }
    this->len = newsize;
  }

  template <typename DType>
  void StorageImplementation<DType>::append(const Scalar &val) {
    Append(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::append(const cytnx_complex128 &val) {
    Append(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::append(const cytnx_complex64 &val) {
    Append(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::append(const cytnx_double &val) {
    Append(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::append(const cytnx_float &val) {
    Append(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::append(const cytnx_uint64 &val) {
    Append(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::append(const cytnx_int64 &val) {
    Append(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::append(const cytnx_uint32 &val) {
    Append(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::append(const cytnx_int32 &val) {
    Append(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::append(const cytnx_uint16 &val) {
    Append(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::append(const cytnx_int16 &val) {
    Append(val);
  }
  template <typename DType>
  void StorageImplementation<DType>::append(const cytnx_bool &val) {
    Append(val);
  }

  template <typename DType>
  boost::intrusive_ptr<Storage_base> StorageImplementation<DType>::real() {
    if constexpr (!is_complex_v<DType>) {
      cytnx_error_msg(true, "[ERROR] Storage.real() can only be called from complex type.%s", "\n");
      return nullptr;
    } else {
      using ValueType = typename DType::value_type;
      if (this->device == Device.cpu) {
        boost::intrusive_ptr<Storage_base> out(new StorageImplementation<ValueType>());
        void *dtmp = malloc(sizeof(ValueType) * this->cap);
        if constexpr (std::is_same_v<DType, cytnx_complex128>) {
          utils_internal::Complexmem_cpu_cdtd(dtmp, this->Mem, this->len, true);

        } else {  // std::is_same_v<DType, cytnx_complex64>
          utils_internal::Complexmem_cpu_cftf(dtmp, this->Mem, this->len, true);
        }
        out->_Init_byptr(dtmp, this->len, this->device, true, this->cap);
        return out;
      } else {
#ifdef UNI_GPU
        boost::intrusive_ptr<Storage_base> out(new StorageImplementation<ValueType>());
        cudaSetDevice(device);
        void *dtmp = utils_internal::cuMalloc_gpu(sizeof(ValueType) * this->cap);
        if constexpr (std::is_same_v<DType, cytnx_complex128>) {
          utils_internal::cuComplexmem_gpu_cdtd(dtmp, this->Mem, this->len, true);

        } else {  // std::is_same_v<DType, cytnx_complex64>
          utils_internal::cuComplexmem_gpu_cftf(dtmp, this->Mem, this->len, true);
        }
        out->_Init_byptr(dtmp, this->len, this->device, true, this->cap);
        return out;
#else
        cytnx_error_msg(
          1, "%s",
          "[ERROR][Internal] Storage.to_. the Storage is as GPU but without CUDA support.");
        return nullptr;
#endif
      }
    }
  }

  template <typename DType>
  boost::intrusive_ptr<Storage_base> StorageImplementation<DType>::imag() {
    if constexpr (!is_complex_v<DType>) {
      cytnx_error_msg(true, "[ERROR] Storage.real() can only be called from complex type.%s", "\n");
      return nullptr;
    } else {
      using ValueType = typename DType::value_type;
      if (this->device == Device.cpu) {
        boost::intrusive_ptr<Storage_base> out(new StorageImplementation<ValueType>());
        void *dtmp = malloc(sizeof(ValueType) * this->cap);
        if constexpr (std::is_same_v<DType, cytnx_complex128>) {
          utils_internal::Complexmem_cpu_cdtd(dtmp, this->Mem, this->len, false);

        } else {  // std::is_same_v<DType, cytnx_complex64>
          utils_internal::Complexmem_cpu_cftf(dtmp, this->Mem, this->len, false);
        }
        out->_Init_byptr(dtmp, this->len, this->device, true, this->cap);
        return out;
      } else {
#ifdef UNI_GPU
        boost::intrusive_ptr<Storage_base> out(new StorageImplementation<ValueType>());
        cudaSetDevice(device);
        void *dtmp = utils_internal::cuMalloc_gpu(sizeof(ValueType) * this->cap);
        if constexpr (std::is_same_v<DType, cytnx_complex128>) {
          utils_internal::cuComplexmem_gpu_cdtd(dtmp, this->Mem, this->len, false);

        } else {  // std::is_same_v<DType, cytnx_complex64>
          utils_internal::cuComplexmem_gpu_cftf(dtmp, this->Mem, this->len, false);
        }
        out->_Init_byptr(dtmp, this->len, this->device, true, this->cap);
        return out;
#else
        cytnx_error_msg(
          1, "%s",
          "[ERROR][Internal] Storage.to_. the Storage is as GPU but without CUDA support.");
        return nullptr;
#endif
      }
    }
  }

  template <typename DType>
  Scalar StorageImplementation<DType>::get_item(const cytnx_uint64 &idx) const {
    return Scalar(this->at<DType>(idx));
  }

  template <typename DType>
  void StorageImplementation<DType>::set_item(const cytnx_uint64 &idx, const Scalar &val) {
    SetItem(idx, val);
  }
  template <typename DType>
  void StorageImplementation<DType>::set_item(const cytnx_uint64 &idx,
                                              const cytnx_complex128 &val) {
    SetItem(idx, val);
  }
  template <typename DType>
  void StorageImplementation<DType>::set_item(const cytnx_uint64 &idx, const cytnx_complex64 &val) {
    SetItem(idx, val);
  }
  template <typename DType>
  void StorageImplementation<DType>::set_item(const cytnx_uint64 &idx, const cytnx_double &val) {
    SetItem(idx, val);
  }
  template <typename DType>
  void StorageImplementation<DType>::set_item(const cytnx_uint64 &idx, const cytnx_float &val) {
    SetItem(idx, val);
  }
  template <typename DType>
  void StorageImplementation<DType>::set_item(const cytnx_uint64 &idx, const cytnx_int64 &val) {
    SetItem(idx, val);
  }
  template <typename DType>
  void StorageImplementation<DType>::set_item(const cytnx_uint64 &idx, const cytnx_uint64 &val) {
    SetItem(idx, val);
  }
  template <typename DType>
  void StorageImplementation<DType>::set_item(const cytnx_uint64 &idx, const cytnx_int32 &val) {
    SetItem(idx, val);
  }
  template <typename DType>
  void StorageImplementation<DType>::set_item(const cytnx_uint64 &idx, const cytnx_uint32 &val) {
    SetItem(idx, val);
  }
  template <typename DType>
  void StorageImplementation<DType>::set_item(const cytnx_uint64 &idx, const cytnx_int16 &val) {
    SetItem(idx, val);
  }
  template <typename DType>
  void StorageImplementation<DType>::set_item(const cytnx_uint64 &idx, const cytnx_uint16 &val) {
    SetItem(idx, val);
  }
  template <typename DType>
  void StorageImplementation<DType>::set_item(const cytnx_uint64 &idx, const cytnx_bool &val) {
    SetItem(idx, val);
  }

  template <typename DType>
  template <typename OtherDType>
  void StorageImplementation<DType>::Fill(const OtherDType &value) {
    if constexpr (!std::is_constructible_v<DType, OtherDType>) {
      cytnx_error_msg(true, "[ERROR] cannot fill %s value into %s container",
                      Type.getname(Type.cy_typeid(OtherDType())).c_str(),
                      Type.getname(Type.cy_typeid(DType())).c_str());
      return;
    } else {
      if (this->device == Device.cpu) {
        utils_internal::FillCpu(this->Mem, static_cast<DType>(value), this->len);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(this->device));
        utils_internal::FillGpu(this->Mem, static_cast<DType>(value), this->len);
#else
        cytnx_error_msg(true, "[ERROR][fill] fatal internal, %s",
                        "storage is on gpu without CUDA support\n");
#endif
      }
    }
  }

  template <typename DType>
  template <typename OtherDType>
  void StorageImplementation<DType>::Append(const OtherDType &value) {
    if constexpr (std::is_constructible_v<DType, OtherDType>) {
      if (this->len == this->cap) {
        this->resize(this->len + 1);
      } else {
        ++this->len;
      }
      this->at<DType>(this->len - 1) = value;
    } else {
      cytnx_error_msg(true, "[ERROR] cannot append %s value into %s container",
                      Type.getname(Type.cy_typeid(OtherDType())).c_str(),
                      Type.getname(Type.cy_typeid(DType())).c_str());
    }
  }

  template <typename DType>
  void StorageImplementation<DType>::Append(const Scalar &value) {
    if (this->len == this->cap) {
      this->resize(this->len + 1);
    } else {
      ++this->len;
    }
    if constexpr (is_same_v<DType, cytnx_complex128>) {
      this->at<DType>(this->len - 1) = complex128(value);
    } else if constexpr (is_same_v<DType, cytnx_complex64>) {
      this->at<DType>(this->len - 1) = complex64(value);
    } else {
      this->at<DType>(this->len - 1) = static_cast<DType>(value);
    }
  }

  template <typename DType>
  template <typename OtherDType>
  void StorageImplementation<DType>::SetItem(cytnx_uint64 index, const OtherDType &value) {
    if constexpr (std::is_constructible_v<DType, OtherDType>) {
      this->at<DType>(index) = value;
    } else {
      cytnx_error_msg(true, "[ERROR] cannot set %s to %s",
                      Type.getname(Type.cy_typeid(OtherDType())).c_str(),
                      Type.getname(Type.cy_typeid(DType())).c_str());
    }
  }

  template <typename DType>
  void StorageImplementation<DType>::SetItem(cytnx_uint64 index, const Scalar &value) {
    if constexpr (is_same_v<DType, cytnx_complex128>) {
      this->at<DType>(index) = complex128(value);
    } else if constexpr (is_same_v<DType, cytnx_complex64>) {
      this->at<DType>(index) = complex64(value);
    } else {
      this->at<DType>(index) = static_cast<DType>(value);
    }
  }

  template class StorageImplementation<cytnx_complex128>;
  template class StorageImplementation<cytnx_complex64>;
  template class StorageImplementation<cytnx_double>;
  template class StorageImplementation<cytnx_float>;
  template class StorageImplementation<cytnx_uint64>;
  template class StorageImplementation<cytnx_int64>;
  template class StorageImplementation<cytnx_uint32>;
  template class StorageImplementation<cytnx_int32>;
  template class StorageImplementation<cytnx_uint16>;
  template class StorageImplementation<cytnx_int16>;
  template class StorageImplementation<cytnx_bool>;
}  // namespace cytnx
