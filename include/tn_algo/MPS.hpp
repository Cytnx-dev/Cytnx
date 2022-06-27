#ifndef _H_MPS_
#define _H_MPS_

#include "cytnx_error.hpp"
#include "Device.hpp"
#include "intrusive_ptr_base.hpp"
#include "UniTensor.hpp"
#include <iostream>
#include <fstream>

#include "utils/vec_clone.hpp"
//#include "utils/dynamic_arg_resolver.hpp"
//#include "linalg.hpp"
#include "Accessor.hpp"
#include <vector>
#include <initializer_list>
#include <string>
#include "Scalar.hpp"

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

      virtual void _save_dispatch(std::fstream &f);
      virtual void _load_dispatch(std::fstream &f);
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

      void _save_dispatch(std::fstream &f);
      void _load_dispatch(std::fstream &f);
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
      void _save_dispatch(std::fstream &f);
      void _load_dispatch(std::fstream &f);
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

      ///@cond
      void _Save(std::fstream &f) const;
      void _Load(std::fstream &f);
      ///@endcond

      void Save(const std::string &fname) const;
      void Save(const char *fname) const;

      static MPS Load(const std::string &fname);
      static MPS Load(const char *fname);
    };

    std::ostream &operator<<(std::ostream &os, const MPS &in);

  }  // namespace tn_algo
}  // namespace cytnx

#endif
