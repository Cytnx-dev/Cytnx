#include "backend/Storage.hpp"
#include "utils_internal_interface.hpp"

using namespace std;

namespace cytnx {
  void Int16Storage::Init(const unsigned long long &len_in, const int &device,
                          const bool &init_zero) {
    // cout << "Int16.init" << endl;
    this->len = len_in;

    // check:
    // cytnx_error_msg(len_in < 1, "%s", "[ERROR] cannot init a Storage with zero element");
    this->dtype = Type.Int16;

    if (this->len % STORAGE_DEFT_SZ) {
      this->cap = ((unsigned long long)((this->len) / STORAGE_DEFT_SZ) + 1) * STORAGE_DEFT_SZ;
    } else {
      this->cap = this->len;
    }

    if (device == Device.cpu) {
      if (init_zero)
        this->Mem = utils_internal::Calloc_cpu(this->cap, sizeof(cytnx_int16));
      else
        this->Mem = utils_internal::Malloc_cpu(this->cap * sizeof(cytnx_int16));
    } else {
#ifdef UNI_GPU
      cytnx_error_msg(device >= Device.Ngpus, "%s", "[ERROR] invalid device.");
      cudaSetDevice(device);
      // this->Mem = utils_internal::cuMalloc_gpu(this->cap*sizeof(cytnx_int16));
      this->Mem = utils_internal::cuCalloc_gpu(this->cap, sizeof(cytnx_int16));
#else
      cytnx_error_msg(1, "%s", "[ERROR] cannot init a Storage on gpu without CUDA support.");
#endif
    }
    this->device = device;
  }

  void Int16Storage::_Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device,
                                 const bool &iscap, const unsigned long long &cap_in) {
    this->Mem = rawptr;
    this->len = len_in;
    if (iscap) {
      this->cap = cap_in;
    } else {
      this->cap = len_in;
    }
    cytnx_error_msg(this->cap % STORAGE_DEFT_SZ != 0,
                    "[ERROR] _Init_by_ptr cannot have not %dx cap_in.", STORAGE_DEFT_SZ);

    if (User_debug) {
      cytnx_error_msg(len_in < 1, "%s", "[ERROR] _Init_by_ptr cannot have len_in < 1.");
      cytnx_error_msg(this->cap < this->len, "%s",
                      "[ERROR] _Init_by_ptr cannot have capacity < size.");
    }
    this->dtype = Type.Int16;
    this->device = device;
  }

  boost::intrusive_ptr<Storage_base> Int16Storage::_create_new_sametype() {
    boost::intrusive_ptr<Storage_base> out(new Int16Storage());
    return out;
  }

  boost::intrusive_ptr<Storage_base> Int16Storage::clone() {
    boost::intrusive_ptr<Storage_base> out(new Int16Storage());
    out->Init(this->len, this->device);
    if (this->device == Device.cpu) {
      memcpy(out->Mem, this->Mem, sizeof(cytnx_int16) * this->len);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      checkCudaErrors(
        cudaMemcpy(out->Mem, this->Mem, sizeof(cytnx_int16) * this->len, cudaMemcpyDeviceToDevice));
#else
      cytnx_error_msg(1, "%s", "[ERROR] cannot clone a Storage on gpu without CUDA support.");
#endif
    }
    return out;
  }

  void Int16Storage::Move_memory_(const std::vector<cytnx_uint64> &old_shape,
                                  const std::vector<cytnx_uint64> &mapper,
                                  const std::vector<cytnx_uint64> &invmapper) {
    boost::intrusive_ptr<Storage_base> tmp(this);
    if (this->device == Device.cpu) {
      utils_internal::Movemem_cpu_i16(tmp, old_shape, mapper, invmapper, 1);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      utils_internal::cuMovemem_gpu_i16(tmp, old_shape, mapper, invmapper, 1);
#else
      cytnx_error_msg(1, "%s", "[ERROR][Internal] try to call GPU section without CUDA support");
#endif
    }
  }

  boost::intrusive_ptr<Storage_base> Int16Storage::Move_memory(
    const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper,
    const std::vector<cytnx_uint64> &invmapper) {
    boost::intrusive_ptr<Storage_base> tmp(this);
    if (this->device == Device.cpu) {
      return utils_internal::Movemem_cpu_i16(tmp, old_shape, mapper, invmapper, 0);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      return utils_internal::cuMovemem_gpu_i16(tmp, old_shape, mapper, invmapper, 0);
#else
      cytnx_error_msg(1, "%s", "[ERROR][Internal] try to call GPU section without CUDA support");
      return nullptr;
#endif
    }
  }
  void Int16Storage::to_(const int &device) {
    if (this->device != device) {
      if (this->device == Device.cpu) {
// here, cpu->gpu with gid=device
#ifdef UNI_GPU
        cytnx_error_msg(device >= Device.Ngpus, "%s", "[ERROR] invalid device.");
        cudaSetDevice(device);
        void *dtmp = utils_internal::cuMalloc_gpu(sizeof(cytnx_int16) * this->cap);
        checkCudaErrors(
          cudaMemcpy(dtmp, this->Mem, sizeof(cytnx_int16) * this->len, cudaMemcpyHostToDevice));
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
          void *htmp = malloc(sizeof(cytnx_int16) * this->cap);
          checkCudaErrors(
            cudaMemcpy(htmp, this->Mem, sizeof(cytnx_int16) * this->len, cudaMemcpyDeviceToHost));
          cudaFree(this->Mem);
          this->Mem = htmp;
          this->device = device;
        } else {
          // here, gpu->gpu
          cytnx_error_msg(device >= Device.Ngpus, "%s", "[ERROR] invalid device.");
          cudaSetDevice(device);
          void *dtmp = utils_internal::cuMalloc_gpu(sizeof(cytnx_int16) * this->cap);
          checkCudaErrors(
            cudaMemcpyPeer(dtmp, device, this->Mem, this->device, sizeof(cytnx_int16) * this->len));
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
  boost::intrusive_ptr<Storage_base> Int16Storage::to(const int &device) {
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
        void *dtmp = utils_internal::cuMalloc_gpu(sizeof(cytnx_int16) * this->cap);
        checkCudaErrors(
          cudaMemcpy(dtmp, this->Mem, sizeof(cytnx_int16) * this->len, cudaMemcpyHostToDevice));
        boost::intrusive_ptr<Storage_base> out(new Int16Storage());
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
          void *htmp = malloc(sizeof(cytnx_int16) * this->cap);
          checkCudaErrors(
            cudaMemcpy(htmp, this->Mem, sizeof(cytnx_int16) * this->len, cudaMemcpyDeviceToHost));
          boost::intrusive_ptr<Storage_base> out(new Int16Storage());
          out->_Init_byptr(htmp, this->len, device, true, this->cap);
          return out;
        } else {
          // here, gpu->gpu
          cytnx_error_msg(device >= Device.Ngpus, "%s", "[ERROR] invalid device.");
          cudaSetDevice(device);
          void *dtmp = utils_internal::cuMalloc_gpu(sizeof(cytnx_int16) * this->cap);
          checkCudaErrors(
            cudaMemcpyPeer(dtmp, device, this->Mem, this->device, sizeof(cytnx_int16) * this->len));
          boost::intrusive_ptr<Storage_base> out(new Int16Storage());
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

  void Int16Storage::PrintElem_byShape(std::ostream &os, const std::vector<cytnx_uint64> &shape,
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
      sprintf(buffer, " (%d", shape[0]);
      os << std::string(buffer);
      for (cytnx_int32 i = 1; i < shape.size(); i++) {
        sprintf(buffer, ",%d", shape[i]);
        os << std::string(buffer);
      }
      os << ")" << std::endl;

      // temporary move to cpu for printing.
      if (this->device != Device.cpu) {
        this->to_(Device.cpu);
      }

      std::vector<cytnx_uint64> stk(shape.size(), 0), stk2;

      cytnx_uint64 s;
      cytnx_int16 *elem_ptr_ = static_cast<cytnx_int16 *>(this->Mem);

      if (mapper.size() == 0) {
        cytnx_uint64 cnt = 0;
        while (1) {
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
            sprintf(buffer, "%+5d ", elem_ptr_[cnt]);
            os << std::string(buffer);
            cnt++;
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

        while (1) {
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
            sprintf(buffer, "%+5d ", elem_ptr_[RealMemPos]);
            os << std::string(buffer);
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
              os << string(buffer);
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

  void Int16Storage::print_elems() {
    char *buffer = (char *)malloc(sizeof(char) * 256);
    cytnx_int16 *elem_ptr_ = static_cast<cytnx_int16 *>(this->Mem);
    cout << "[ ";
    for (unsigned long long cnt = 0; cnt < this->len; cnt++) {
      sprintf(buffer, "%+5d ", elem_ptr_[cnt]);
      cout << std::string(buffer);
    }
    cout << " ]" << endl;
    free(buffer);
  }

  void Int16Storage::fill(const cytnx_complex128 &val) {
    cytnx_error_msg(true, "[ERROR]%s", " cannot fill complex value into real container");
  }
  void Int16Storage::fill(const cytnx_complex64 &val) {
    cytnx_error_msg(true, "[ERROR]%s", " cannot fill complex value into real container");
  }
  void Int16Storage::fill(const cytnx_double &val) {
    cytnx_int16 tmp = val;
    if (this->device == Device.cpu) {
      utils_internal::Fill_cpu_i16(this->Mem, (void *)(&tmp), this->len);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      utils_internal::cuFill_gpu_i16(this->Mem, (void *)(&tmp), this->len);
#else
      cytnx_error_msg(true, "[ERROR][fill] fatal internal, %s",
                      "storage is on gpu without CUDA support\n");
#endif
    }
  }
  void Int16Storage::fill(const cytnx_float &val) {
    cytnx_int16 tmp = val;
    if (this->device == Device.cpu) {
      utils_internal::Fill_cpu_i16(this->Mem, (void *)(&tmp), this->len);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      utils_internal::cuFill_gpu_i16(this->Mem, (void *)(&tmp), this->len);
#else
      cytnx_error_msg(true, "[ERROR][fill] fatal internal, %s",
                      "storage is on gpu without CUDA support\n");
#endif
    }
  }
  void Int16Storage::fill(const cytnx_int64 &val) {
    cytnx_int16 tmp = val;
    if (this->device == Device.cpu) {
      utils_internal::Fill_cpu_i16(this->Mem, (void *)(&tmp), this->len);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      utils_internal::cuFill_gpu_i16(this->Mem, (void *)(&tmp), this->len);
#else
      cytnx_error_msg(true, "[ERROR][fill] fatal internal, %s",
                      "storage is on gpu without CUDA support\n");
#endif
    }
  }
  void Int16Storage::fill(const cytnx_uint64 &val) {
    cytnx_int16 tmp = val;
    if (this->device == Device.cpu) {
      utils_internal::Fill_cpu_i16(this->Mem, (void *)(&tmp), this->len);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      utils_internal::cuFill_gpu_i16(this->Mem, (void *)(&tmp), this->len);
#else
      cytnx_error_msg(true, "[ERROR][fill] fatal internal, %s",
                      "storage is on gpu without CUDA support\n");
#endif
    }
  }
  void Int16Storage::fill(const cytnx_int32 &val) {
    cytnx_int16 tmp = val;
    if (this->device == Device.cpu) {
      utils_internal::Fill_cpu_i16(this->Mem, (void *)(&tmp), this->len);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      utils_internal::cuFill_gpu_i16(this->Mem, (void *)(&tmp), this->len);
#else
      cytnx_error_msg(true, "[ERROR][fill] fatal internal, %s",
                      "storage is on gpu without CUDA support\n");
#endif
    }
  }
  void Int16Storage::fill(const cytnx_uint32 &val) {
    cytnx_int16 tmp = val;
    if (this->device == Device.cpu) {
      utils_internal::Fill_cpu_i16(this->Mem, (void *)(&tmp), this->len);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      utils_internal::cuFill_gpu_i16(this->Mem, (void *)(&tmp), this->len);
#else
      cytnx_error_msg(true, "[ERROR][fill] fatal internal, %s",
                      "storage is on gpu without CUDA support\n");
#endif
    }
  }
  void Int16Storage::fill(const cytnx_uint16 &val) {
    cytnx_int16 tmp = val;
    if (this->device == Device.cpu) {
      utils_internal::Fill_cpu_i16(this->Mem, (void *)(&tmp), this->len);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      utils_internal::cuFill_gpu_i16(this->Mem, (void *)(&tmp), this->len);
#else
      cytnx_error_msg(true, "[ERROR][fill] fatal internal, %s",
                      "storage is on gpu without CUDA support\n");
#endif
    }
  }
  void Int16Storage::fill(const cytnx_int16 &val) {
    if (this->device == Device.cpu) {
      utils_internal::Fill_cpu_i16(this->Mem, (void *)(&val), this->len);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      utils_internal::cuFill_gpu_i16(this->Mem, (void *)(&val), this->len);
#else
      cytnx_error_msg(true, "[ERROR][fill] fatal internal, %s",
                      "storage is on gpu without CUDA support\n");
#endif
    }
  }
  void Int16Storage::fill(const cytnx_bool &val) {
    cytnx_int16 tmp = val;
    if (this->device == Device.cpu) {
      utils_internal::Fill_cpu_i16(this->Mem, (void *)(&tmp), this->len);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      utils_internal::cuFill_gpu_i16(this->Mem, (void *)(&tmp), this->len);
#else
      cytnx_error_msg(true, "[ERROR][fill] fatal internal, %s",
                      "storage is on gpu without CUDA support\n");
#endif
    }
  }

  void Int16Storage::set_zeros() {
    if (this->device == Device.cpu) {
      utils_internal::SetZeros(this->Mem, sizeof(cytnx_int16) * this->len);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device));
      utils_internal::cuSetZeros(this->Mem, sizeof(cytnx_int16) * this->len);
#else
      cytnx_error_msg(1, "[ERROR][set_zeros] fatal, the storage is on gpu without CUDA support.%s",
                      "\n");
#endif
    }
  }

  void Int16Storage::resize(const cytnx_uint64 &newsize) {
    // cytnx_error_msg(newsize < 1,"[ERROR]resize should have size > 0%s","\n");

    if (newsize > this->cap) {
      if (newsize % STORAGE_DEFT_SZ) {
        this->cap = ((unsigned long long)((newsize) / STORAGE_DEFT_SZ) + 1) * STORAGE_DEFT_SZ;
      } else {
        this->cap = newsize;
      }
      if (this->device == Device.cpu) {
        void *htmp = calloc(this->cap, sizeof(cytnx_int16));
        memcpy(htmp, this->Mem, sizeof(cytnx_int16) * this->len);
        free(this->Mem);
        this->Mem = htmp;
      } else {
#ifdef UNI_GPU
        cytnx_error_msg(device >= Device.Ngpus, "%s", "[ERROR] invalid device.");
        cudaSetDevice(device);
        void *dtmp = utils_internal::cuCalloc_gpu(this->cap, sizeof(cytnx_int16));
        checkCudaErrors(
          cudaMemcpyPeer(dtmp, device, this->Mem, this->device, sizeof(cytnx_int16) * this->len));
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

  void Int16Storage::append(const Scalar &val) {
    if (this->len + 1 > this->cap) {
      this->resize(this->len + 1);
    } else {
      this->len += 1;
    }
    this->at<cytnx_int16>(this->len - 1) = cytnx_int16(val);
  }
  void Int16Storage::append(const cytnx_complex128 &val) {
    cytnx_error_msg(true, "[ERROR]%s", " cannot append complex value into real container");
  }
  void Int16Storage::append(const cytnx_complex64 &val) {
    cytnx_error_msg(true, "[ERROR]%s", " cannot append complex value into real container");
  }
  void Int16Storage::append(const cytnx_double &val) {
    if (this->len + 1 > this->cap) {
      this->resize(this->len + 1);
    } else {
      this->len += 1;
    }
    this->at<cytnx_int16>(this->len - 1) = val;
  }
  void Int16Storage::append(const cytnx_float &val) {
    if (this->len + 1 > this->cap) {
      this->resize(this->len + 1);
    } else {
      this->len += 1;
    }
    this->at<cytnx_int16>(this->len - 1) = val;
  }
  void Int16Storage::append(const cytnx_int64 &val) {
    if (this->len + 1 > this->cap) {
      this->resize(this->len + 1);
    } else {
      this->len += 1;
    }
    this->at<cytnx_int16>(this->len - 1) = val;
  }
  void Int16Storage::append(const cytnx_int32 &val) {
    if (this->len + 1 > this->cap) {
      this->resize(this->len + 1);
    } else {
      this->len += 1;
    }
    this->at<cytnx_int16>(this->len - 1) = val;
  }
  void Int16Storage::append(const cytnx_int16 &val) {
    if (this->len + 1 > this->cap) {
      this->resize(this->len + 1);
    } else {
      this->len += 1;
    }
    this->at<cytnx_int16>(this->len - 1) = val;
  }
  void Int16Storage::append(const cytnx_uint64 &val) {
    if (this->len + 1 > this->cap) {
      this->resize(this->len + 1);
    } else {
      this->len += 1;
    }
    this->at<cytnx_int16>(this->len - 1) = val;
  }
  void Int16Storage::append(const cytnx_uint32 &val) {
    if (this->len + 1 > this->cap) {
      this->resize(this->len + 1);
    } else {
      this->len += 1;
    }
    this->at<cytnx_int16>(this->len - 1) = val;
  }
  void Int16Storage::append(const cytnx_uint16 &val) {
    if (this->len + 1 > this->cap) {
      this->resize(this->len + 1);
    } else {
      this->len += 1;
    }
    this->at<cytnx_int16>(this->len - 1) = val;
  }
  void Int16Storage::append(const cytnx_bool &val) {
    if (this->len + 1 > this->cap) {
      this->resize(this->len + 1);
    } else {
      this->len += 1;
    }
    this->at<cytnx_int16>(this->len - 1) = val;
  }

  boost::intrusive_ptr<Storage_base> Int16Storage::real() {
    cytnx_error_msg(true, "[ERROR] Storage.real() can only be called from complex type.%s", "\n");
  }
  boost::intrusive_ptr<Storage_base> Int16Storage::imag() {
    cytnx_error_msg(true, "[ERROR] Storage.imag() can only be called from complex type.%s", "\n");
  }

  Scalar Int16Storage::get_item(const cytnx_uint64 &idx) const {
    return Scalar(this->at<cytnx_int16>(idx));
  }

  void Int16Storage::set_item(const cytnx_uint64 &idx, const Scalar &val) {
    this->at<cytnx_int16>(idx) = cytnx_int16(val);
  }
  void Int16Storage::set_item(const cytnx_uint64 &idx, const cytnx_complex128 &val) {
    cytnx_error_msg(true, "[ERROR] cannot set complex to real.%s", "\n");
  }
  void Int16Storage::set_item(const cytnx_uint64 &idx, const cytnx_complex64 &val) {
    cytnx_error_msg(true, "[ERROR] cannot set complex to real.%s", "\n");
  }
  void Int16Storage::set_item(const cytnx_uint64 &idx, const cytnx_double &val) {
    this->at<cytnx_int16>(idx) = val;
  }
  void Int16Storage::set_item(const cytnx_uint64 &idx, const cytnx_float &val) {
    this->at<cytnx_int16>(idx) = val;
  }
  void Int16Storage::set_item(const cytnx_uint64 &idx, const cytnx_int64 &val) {
    this->at<cytnx_int16>(idx) = val;
  }
  void Int16Storage::set_item(const cytnx_uint64 &idx, const cytnx_uint64 &val) {
    this->at<cytnx_int16>(idx) = val;
  }
  void Int16Storage::set_item(const cytnx_uint64 &idx, const cytnx_int32 &val) {
    this->at<cytnx_int16>(idx) = val;
  }
  void Int16Storage::set_item(const cytnx_uint64 &idx, const cytnx_uint32 &val) {
    this->at<cytnx_int16>(idx) = val;
  }
  void Int16Storage::set_item(const cytnx_uint64 &idx, const cytnx_int16 &val) {
    this->at<cytnx_int16>(idx) = val;
  }
  void Int16Storage::set_item(const cytnx_uint64 &idx, const cytnx_uint16 &val) {
    this->at<cytnx_int16>(idx) = val;
  }
  void Int16Storage::set_item(const cytnx_uint64 &idx, const cytnx_bool &val) {
    this->at<cytnx_int16>(idx) = val;
  }

  // bool Int16Storage::approx_eq(const boost::intrusive_ptr<Storage_base> &rhs,
  //                              const cytnx_double tol) {
  //   boost::intrusive_ptr<Storage_base> _lhs, _rhs;
  //   if (rhs->dtype == this->dtype) {
  //     _lhs = this;
  //     _rhs = rhs;
  //   } else if (rhs->dtype > this->dtype) {
  //     _lhs = this;
  //     _rhs = rhs->astype(this->dtype);
  //   } else {
  //     _lhs = this->astype(rhs->dtype);
  //     _rhs = rhs;
  //   }
  //   if (_rhs->size() != _lhs->size()) {
  //     if (User_debug) std::cout << "different tensor size." << std::endl;
  //     return false;
  //   }
  //   for (cytnx_uint64 i = 0; i < this->len; i++) {
  //     if (_lhs->get_item(i).approx_eq(_rhs->get_item(i), tol) == false) {
  //       if (User_debug)
  //         std::cout << "tensor different at idx:" << i << "\n"
  //                   << "lhs:" << _lhs->get_item(i) << " rhs:" << _rhs->get_item(i) << "\n";
  //       return false;
  //     }
  //   }
  //   return true;
  // }
}  // namespace cytnx
