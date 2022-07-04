#include "tn_algo/MPS.hpp"

using namespace std;
namespace cytnx {
  namespace tn_algo {

    //====================================================
    std::string MPSType_class::getname(const int &mps_type) {
      if (mps_type == this->Void) {
        return std::string("Void (un-initialize MPS");
      } else if (mps_type == this->RegularMPS) {
        return std::string("Regular MPS");
      } else if (mps_type == this->iMPS) {
        return std::string("iMPS");
      } else {
        cytnx_error_msg(true, "%s\n", "[ERROR] invalid mps_type");
        return std::string("");
      }
      // extend more in here!!
    }
    MPSType_class MPSType;
    //===================================================

    Scalar MPS_impl::norm() const {
      cytnx_error_msg(
        true, "[ERROR] MPS_Base should not be called. Please initialize the MPS first.%s", "\n");
      return Scalar(0);
    }

    boost::intrusive_ptr<MPS_impl> MPS_impl::clone() const {
      cytnx_error_msg(
        true, "[ERROR] MPS_Base should not be called. Please initialize the MPS first.%s", "\n");
      return boost::intrusive_ptr<MPS_impl>(new MPS_impl());
    }

    std::ostream &MPS_impl::Print(std::ostream &os) {
      cytnx_error_msg(
        true, "[ERROR] MPS_Base should not be called. Please initialize the MPS first.%s", "\n");
      return os;
    }

    void MPS_impl::Init(const cytnx_uint64 &N, const std::vector<cytnx_uint64> &vphys_dim,
                        const cytnx_uint64 &virt_dim, const cytnx_int64 &dtype) {
      cytnx_error_msg(
        true, "[ERROR] MPS_Base should not be called. Please initialize the MPS first.%s", "\n");
    }
    void MPS_impl::Init_Msector(const cytnx_uint64 &N, const std::vector<cytnx_uint64> &vphys_dim,
                                const cytnx_uint64 &virt_dim,
                                const std::vector<cytnx_int64> &select, const cytnx_int64 &dtype) {
      cytnx_error_msg(
        true, "[ERROR] MPS_Base should not be called. Please initialize the MPS first.%s", "\n");
    }
    /*
    void MPS_impl::Init_prodstate(const std::vector<cytnx_uint64> &phys_dim, const cytnx_uint64
    &virt_dim, const std::vector<std::vector<cytnx_int64> >&state_qnums, const cytnx_int64 &dtype){
        cytnx_error_msg(true,"[ERROR] MPS_Base should not be called. Please initialize the MPS
    first.%s","\n");
    }
    */
    void MPS_impl::Into_Lortho() {
      cytnx_error_msg(
        true, "[ERROR] MPS_Base should not be called. Please initialize the MPS first.%s", "\n");
    }
    void MPS_impl::S_mvleft() {
      cytnx_error_msg(
        true, "[ERROR] MPS_Base should not be called. Please initialize the MPS first.%s", "\n");
    }
    void MPS_impl::S_mvright() {
      cytnx_error_msg(
        true, "[ERROR] MPS_Base should not be called. Please initialize the MPS first.%s", "\n");
    }

    void MPS_impl::_save_dispatch(fstream &f) {
      cytnx_error_msg(
        true, "[ERROR] MPS_Base should not be called. Please initialize the MPS first.%s", "\n");
    }

    void MPS_impl::_load_dispatch(fstream &f) {
      cytnx_error_msg(
        true, "[ERROR] MPS_Base should not be called. Please initialize the MPS first.%s", "\n");
    }

  }  // namespace tn_algo

}  // namespace cytnx
