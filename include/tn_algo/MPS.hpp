#ifndef CYTNX_TN_ALGO_MPS_H_
#define CYTNX_TN_ALGO_MPS_H_

#include <fstream>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

#include "H5Cpp.h"

#include "Accessor.hpp"
#include "Device.hpp"
#include "UniTensor.hpp"
#include "cytnx_error.hpp"
#include "intrusive_ptr_base.hpp"
#include "utils/vec_clone.hpp"

#ifdef BACKEND_TORCH
#else

  #include "backend/Scalar.hpp"

namespace cytnx {
  namespace tn_algo {

    /// @cond
    class MPSType_class {
     public:
      enum : int {
        Void = -99,
        RegularMPS = 0,
        iMPS = 1,
      };
      std::string getname(const int &mps_type);
    };
    extern MPSType_class MPSType;

    class MPS_impl : public intrusive_ptr_base<MPS_impl> {
     private:
     public:
      friend class MPS;

      // std::vector<cytnx_int64> phys_dim;
      cytnx_int64 virt_dim;  // maximum
      cytnx_int64 S_loc;

      int mps_type_id;

      MPS_impl() : mps_type_id(MPSType.Void) {}

      // place holder for the tensors:
      std::vector<UniTensor> _TNs;

      std::vector<UniTensor> &get_data() { return this->_TNs; }

      virtual Scalar norm() const;
      virtual boost::intrusive_ptr<MPS_impl> clone() const;
      virtual std::ostream &Print(std::ostream &os);
      virtual cytnx_uint64 size() { return 0; };
      virtual void Init(const cytnx_uint64 &N, const std::vector<cytnx_uint64> &phys_dim,
                        const cytnx_uint64 &virt_dim, const cytnx_int64 &dtype);
      virtual void Init_Msector(const cytnx_uint64 &N, const std::vector<cytnx_uint64> &phys_dim,
                                const cytnx_uint64 &virt_dim,
                                const std::vector<cytnx_int64> &select, const cytnx_int64 &dtype);
      // virtual void Init_prodstate(const std::vector<cytnx_uint64> &phys_dim, const
      // std::vector<cytnx_uint64> cstate, const cytnx_int64 &dtype);

      // for finite MPS:

      // virtual void Init_prodstate(const std::vector<cytnx_uint64> &phys_dim, const cytnx_uint64
      // &virt_dim, const std::vector<std::vector<cytnx_int64> > &state_qnums, const cytnx_int64
      // &dtype);

      virtual void Into_Lortho();
      virtual void S_mvleft();
      virtual void S_mvright();

      virtual void to_binary_dispatch(std::ostream &f);
      virtual void from_binary_dispatch(std::istream &f, const bool restore_device = true);
    };

    // finite size:
    class RegularMPS : public MPS_impl {
     public:
      // only for this:
      RegularMPS() {
        this->mps_type_id = MPSType.RegularMPS;
        this->S_loc = 0;
        this->virt_dim = -1;
      };

      // specialization:
      std::ostream &Print(std::ostream &os);
      cytnx_uint64 size() { return this->_TNs.size(); };
      void Init(const cytnx_uint64 &N, const std::vector<cytnx_uint64> &phys_dim,
                const cytnx_uint64 &virt_dim, const cytnx_int64 &dtype);
      void Init_Msector(const cytnx_uint64 &N, const std::vector<cytnx_uint64> &phys_dim,
                        const cytnx_uint64 &virt_dim, const std::vector<cytnx_int64> &select,
                        const cytnx_int64 &dtype);
      // void Init_prodstate(const std::vector<cytnx_uint64> &phys_dim, const
      // std::vector<cytnx_uint64> cstate, const cytnx_int64 &dtype);

      // void Init_prodstate(const std::vector<cytnx_uint64> &phys_dim, const cytnx_uint64
      // &virt_dim, const std::vector<std::vector<cytnx_int64> >&state_qnums, const cytnx_int64
      // &dtype);

      void Into_Lortho();
      void S_mvleft();
      void S_mvright();

      Scalar norm() const;
      boost::intrusive_ptr<MPS_impl> clone() const {
        boost::intrusive_ptr<MPS_impl> out(new RegularMPS());
        out->S_loc = this->S_loc;
        out->virt_dim = this->virt_dim;
        out->_TNs = vec_clone(this->_TNs);
        return out;
      }

      void to_binary_dispatch(std::ostream &f);
      void from_binary_dispatch(std::istream &f, const bool restore_device = true);
    };

    // infinite size:
    class iMPS : public MPS_impl {
     public:
      // only for this:
      iMPS() {
        this->mps_type_id = MPSType.iMPS;
        this->virt_dim = -1;
      };

      // specialization:
      std::ostream &Print(std::ostream &os);
      cytnx_uint64 size() { return this->_TNs.size(); };
      void Init(const cytnx_uint64 &N, const std::vector<cytnx_uint64> &phys_dim,
                const cytnx_uint64 &virt_dim, const cytnx_int64 &dtype);
      void Init_Msector(const cytnx_uint64 &N, const std::vector<cytnx_uint64> &phys_dim,
                        const cytnx_uint64 &virt_dim, const std::vector<cytnx_int64> &select,
                        const cytnx_int64 &dtype) {
        cytnx_error_msg(true, "[ERROR][MPS][type=iMPS] cannot call Init_Msector%s", "\n");
      }
      // void Init_prodstate(const std::vector<cytnx_uint64> &phys_dim, const
      // std::vector<cytnx_uint64> cstate, const cytnx_int64 &dtype);
      //     cytnx_error_msg(true,"[ERROR][MPS][type=iMPS] cannot call prodstate%s","\n");
      // }
      void Into_Lortho() {
        cytnx_error_msg(true, "[ERROR][MPS][type=iMPS] cannot call Into_Lortho%s", "\n");
      }
      void S_mvleft() {
        cytnx_error_msg(true, "[ERROR][MPS][type=iMPS] cannot call S_mvleft%s", "\n");
      }
      void S_mvright() {
        cytnx_error_msg(true, "[ERROR][MPS][type=iMPS] cannot call S_mvright%s", "\n");
      }
      boost::intrusive_ptr<MPS_impl> clone() const {
        boost::intrusive_ptr<MPS_impl> out(new RegularMPS());
        out->S_loc = this->S_loc;
        out->virt_dim = this->virt_dim;
        out->_TNs = vec_clone(this->_TNs);
        return out;
      }
      Scalar norm() const;
      void to_binary_dispatch(std::ostream &f);
      void from_binary_dispatch(std::istream &f, const bool restore_device = true);
    };
    ///@endcond

    // API
    class MPS {
     private:
     public:
      ///@cond
      boost::intrusive_ptr<MPS_impl> _impl;
      MPS()
          : _impl(new MPS_impl()){
              // currently default init is RegularMPS;:
            };

      MPS(const cytnx_uint64 &N, const cytnx_uint64 &phys_dim, const cytnx_uint64 &virt_dim,
          const cytnx_int64 &dtype = Type.Double, const cytnx_int64 &mps_type = 0)
          : _impl(new MPS_impl()) {
        this->Init(N, phys_dim, virt_dim, dtype, mps_type);
      };

      MPS(const cytnx_uint64 &N, const std::vector<cytnx_uint64> &vphys_dim,
          const cytnx_uint64 &virt_dim, const cytnx_int64 &dtype = Type.Double,
          const cytnx_int64 &mps_type = 0)
          : _impl(new MPS_impl()) {
        this->Init(N, vphys_dim, virt_dim, dtype, mps_type);
      };

      MPS(const MPS &rhs) { _impl = rhs._impl; }

      MPS &operator=(const MPS &rhs) {
        _impl = rhs._impl;
        return *this;
      }
      ///@endcond

      // Initialization API:
      //-----------------------
      MPS &Init(const cytnx_uint64 &N, const std::vector<cytnx_uint64> &vphys_dim,
                const cytnx_uint64 &virt_dim, const cytnx_int64 &dtype = Type.Double,
                const cytnx_int64 &mps_type = 0) {
        if (mps_type == 0) {
          this->_impl = boost::intrusive_ptr<MPS_impl>(new RegularMPS());
        } else if (mps_type == 1) {
          this->_impl = boost::intrusive_ptr<MPS_impl>(new iMPS());
        } else {
          cytnx_error_msg(true, "[ERROR] invalid MPS type.%s", "\n");
        }
        this->_impl->Init(N, vphys_dim, virt_dim, dtype);
        return *this;
      }
      MPS &Init(const cytnx_uint64 &N, const cytnx_uint64 &phys_dim, const cytnx_uint64 &virt_dim,
                const cytnx_int64 &dtype = Type.Double, const cytnx_int64 &mps_type = 0) {
        std::vector<cytnx_uint64> vphys_dim(N, phys_dim);

        this->Init(N, vphys_dim, virt_dim, dtype);
        return *this;
      }
      //-----------------------

      MPS &Init_Msector(const cytnx_uint64 &N, const std::vector<cytnx_uint64> &vphys_dim,
                        const cytnx_uint64 &virt_dim, const std::vector<cytnx_int64> &select,
                        const cytnx_int64 &dtype = Type.Double, const cytnx_int64 &mps_type = 0) {
        // only the select phys index will have non-zero element.
        if (mps_type == 0) {
          this->_impl = boost::intrusive_ptr<MPS_impl>(new RegularMPS());
        } else if (mps_type == 1) {
          this->_impl = boost::intrusive_ptr<MPS_impl>(new iMPS());
        } else {
          cytnx_error_msg(true, "[ERROR] invalid MPS type.%s", "\n");
        }
        this->_impl->Init_Msector(N, vphys_dim, virt_dim, select, dtype);
        return *this;
      }

      /*
      MPS& Init_prodstate(const std::vector<cytnx_uint64> &phys_dim, const std::vector<cytnx_uint64>
      cstate, const cytnx_int64 &dtype){
          // only the select phys index will have non-zero element.
          if(mps_type==0){
              this->_impl =boost::intrusive_ptr<MPS_impl>(new RegularMPS());
          }else if(mps_type==1){
              this->_impl =boost::intrusive_ptr<MPS_impl>(new iMPS());
          }else{
              cytnx_error_msg(true,"[ERROR] invalid MPS type.%s","\n");
          }
          this->_impl->Init_prodstate(phys_dim, cstate, dtype);
          return *this;
      }
      */

      cytnx_uint64 size() { return this->_impl->size(); }

      int mps_type() const { return this->_impl->mps_type_id; }
      std::string mps_type_str() const { return MPSType.getname(this->_impl->mps_type_id); }

      MPS clone() const {
        MPS out;
        out._impl = this->_impl->clone();
        return out;
      }

      std::vector<UniTensor> &data() { return this->_impl->get_data(); };

      MPS &Into_Lortho() {
        this->_impl->Into_Lortho();
        return *this;
      }
      MPS &S_mvleft() {
        this->_impl->S_mvleft();
        return *this;
      }
      MPS &S_mvright() {
        this->_impl->S_mvright();
        return *this;
      }

      Scalar norm() const { return this->_impl->norm(); }

      cytnx_int64 phys_dim(const cytnx_int64 &idx) { return this->_impl->_TNs[idx].shape()[1]; }

      cytnx_int64 &virt_dim() { return this->_impl->virt_dim; }

      cytnx_int64 &S_loc() { return this->_impl->S_loc; }

      /**
       * @brief Save MPS to file
       * @param[in] fname file name
       * @details Save the MPS to a file. The file ending should be one of ".h5", ".hdf5", ".H5",
       * ".HDF5", ".hdf" to save in HDF5 file format. Otherwise, a binary file format is used.
       * @note The common file ending for saving a MPS in binary format is ".cymps".
       * @warning HDF5 file format is strongly recommended for compatibility with other libraries,
       * readability, and future-proofing.
       * @see Load(const std::string &fname, const bool restore_device)
       */
      void Save(const std::string &fname) const;
      // @see Save(const std::string &fname) const
      void Save(const char *fname) const;

      /**
       * @brief Load MPS from file and create new instance
       * @param fname[in] file name
       * @param[in] restore_device whether to try restoring the device on which the data is stored;
       * if false, the data will be kept on the CPU. Use .to_() to move it to the target device
       * after loading.
       * @pre The file must be a MPS object which is saved by cytnx::MPS::Save.
       * @note This function creates a new MPS and keeps the original MPS unchanged. See \link
       * Load_(const std::string &fname, const bool restore_device) Load_() \endlink for loading the
       * MPS to the current MPS.
       * @details For HDF5 file format, one of the file endings ".h5", ".hdf5", ".H5", ".HDF5",
       * ".hdf" is expected. For binary format, the common file ending for a MPS is ".cymps".
       */
      static MPS Load(const std::string &fname, const bool restore_device = true);
      // @see Load(const std::string &fname)
      static MPS Load(const char *fname, const bool restore_device = true);

      /**
       * @brief Load MPS from file and overwrite current instance
       * @note This function overwrites the existing MPS. See \link Load(const std::string &fname,
       * const bool restore_device) Load() \endlink for creating a new MPS.
       * @see Load(const std::string &fname, const bool restore_device)
       */
      void Load_(const std::string &fname, const bool restore_device = true);
      // @see Load_(const std::string &fname, const bool restore_device)
      void Load_(const char *fname, const bool restore_device = true);

      /**
       * @brief Save MPS to HDF5 file
       * @param[in] location the HDF5 group where the MPS will be saved.
       * @param[in] name the name of the MPS in the HDF5 file.
       * @warning This function is only available in C++. Use \link Save(const std::string &fname)
       * Save() \endlink for saving to file in C++ or Python.
       * @see from_hdf5(H5::Group &location, const std::string &name, const bool restore_device)
       */
      void to_hdf5(H5::Group &location, const std::string &name = "MPS") const;
      /**
       * @brief Load MPS from HDF5 file (inline)
       * @param[in] location the HDF5 group where the MPS will be loaded from.
       * @param[in] name the name of the MPS in the HDF5 file.
       * @param[in] restore_device whether to try restoring the device on which the data is stored;
       * if false, the data will be kept on the CPU. Use .to_() to move it to the target device
       * after loading.
       * @warning This function is only available in C++. Use \link Load(const std::string &fname,
       * const bool restore_device) Load() \endlink for loading from file in C++ or Python.
       * @see to_hdf5(H5::Group &location, const std::string &name) const
       */
      void from_hdf5(H5::Group &location, const std::string &name = "MPS",
                     const bool restore_device = true);

      /**
       * @brief Save MPS to binary file
       * @param[in] f the output stream where the MPS will be saved.
       * @warning This function is only available in C++. In Python, use pickle for the same binary
       * file format. Use \link Save(const std::string &fname) Save() \endlink for saving to file in
       * C++ or Python.
       * @see from_binary(std::istream &f, const bool restore_device)
       */
      void to_binary(std::ostream &f) const;
      /**
       * @brief Load MPS from binary file
       * @param[in] f the input stream from which the MPS will be loaded.
       * @param[in] restore_device whether to try restoring the device on which the data is stored;
       * if false, the data will be kept on the CPU. Use .to_() to move it to the target device
       * after loading.
       * @warning This function is only available in C++. In Python, use pickle for the same binary
       * file format. Use \link Load(const std::string &fname, const bool restore_device) Load()
       * \endlink for loading from file in C++ or Python.
       * @see to_binary(std::ostream &f) const
       */
      void from_binary(std::istream &f, const bool restore_device = true);
    };

    std::ostream &operator<<(std::ostream &os, const MPS &in);

  }  // namespace tn_algo
}  // namespace cytnx

#endif  // BACKEND_TORCH

#endif  // CYTNX_TN_ALGO_MPS_H_
