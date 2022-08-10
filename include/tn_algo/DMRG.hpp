#ifndef _H_DMRG_
#define _H_DMRG_

#include "cytnx_error.hpp"
#include "Device.hpp"
#include "intrusive_ptr_base.hpp"
#include "UniTensor.hpp"
#include <iostream>
#include <fstream>
#include "utils/vec_clone.hpp"
#include "Accessor.hpp"
#include <vector>
#include <initializer_list>
#include <string>
#include "Scalar.hpp"
#include "tn_algo/MPS.hpp"
#include "tn_algo/MPO.hpp"

namespace cytnx {
  namespace tn_algo {
    ///@cond
    class DMRG_impl : public intrusive_ptr_base<DMRG_impl> {
     private:
     public:
      MPS mps;
      MPO mpo;

      // for getting excited states:
      std::vector<MPS> ortho_mps;
      double weight;

      // iterative solver param:
      cytnx_int64 maxit;
      cytnx_int64 krydim;

      // environ:
      std::vector<UniTensor> LR;
      std::vector<std::vector<UniTensor>> hLRs;  // excited states

      friend class MPS;
      friend class MPO;

      void initialize();
      Scalar sweep(const bool &verbose, const cytnx_int64 &maxit, const cytnx_int64 &krydim);
      Scalar sweepv2(const bool &verbose, const cytnx_int64 &maxit, const cytnx_int64 &krydim);
    };
    ///@endcond

    // API
    class DMRG {
     private:
     public:
      ///@cond
      boost::intrusive_ptr<DMRG_impl> _impl;
      DMRG(MPO mpo, MPS mps, std::vector<MPS> ortho_mps = {}, const double &weight = 30)
          : _impl(new DMRG_impl()) {
        // currently default init is DMRG_impl;

        // mpo and mps:
        this->_impl->mpo = mpo;
        this->_impl->mps = mps;

        // for getting excited states:
        this->_impl->ortho_mps = ortho_mps;
        this->_impl->weight = weight;
      };

      DMRG(const DMRG &rhs) { _impl = rhs._impl; }
      ///@endcond

      DMRG &operator=(const DMRG &rhs) {
        _impl = rhs._impl;
        return *this;
      }

      DMRG &initialize() {
        this->_impl->initialize();
        return *this;
      }

      // return the current energy
      Scalar sweep(const bool &verbose = false, const cytnx_int64 &maxit = 4000,
                   const cytnx_int64 &krydim = 4) {
        return this->_impl->sweep(verbose, maxit, krydim);
      }

      // return the current energy
      Scalar sweepv2(const bool &verbose = false, const cytnx_int64 &maxit = 4000,
                     const cytnx_int64 &krydim = 4) {
        return this->_impl->sweepv2(verbose, maxit, krydim);
      }
    };

  }  // namespace tn_algo
}  // namespace cytnx

#endif
