#include "tn_algo/MPS.hpp"
#include "random.hpp"
#include <cmath>
#include <algorithm>
using namespace std;
namespace cytnx {
  namespace tn_algo {
    std::ostream &iMPS::Print(std::ostream &os) {
      os << "MPS type : "
         << "[iMPS]" << endl;
      os << "Size : " << this->_TNs.size() << endl;
      os << "physBD dim :\n";

      // print Sloc indicator:
      if (this->S_loc == -1) {
        os << ".[";
      } else {
        os << " [";
      }
      for (int i = 0; i < this->_TNs.size(); i++) {
        os << " ";
        if (this->S_loc == i)
          os << "'" << this->_TNs[i].shape()[1] << "'";
        else
          os << this->_TNs[i].shape()[1];
      }
      if (this->S_loc == this->_TNs.size()) {
        os << " ].\n";
      } else {
        os << "] \n";
      }
      os << "virtBD dim : " << this->virt_dim << endl;
      os << endl;
      return os;
    }

    void iMPS::Init(const cytnx_uint64 &N, const std::vector<cytnx_uint64> &vphys_dim,
                    const cytnx_uint64 &virt_dim, const cytnx_int64 &dtype) {
      // checking:
      cytnx_error_msg(N == 0, "[ERROR][iMPS] number of site N cannot be ZERO.%s", "\n");
      cytnx_error_msg(N != vphys_dim.size(), "[ERROR][iMPS] vphys_dim.size() should equal to N.%s",
                      "\n");
      cytnx_error_msg(dtype != Type.Double,
                      "[ERROR][iMPS] MPS currently only support Double type.%s", "\n");

      this->virt_dim = virt_dim;

      const cytnx_uint64 &chi = virt_dim;

      this->_TNs.resize(N);

      for (cytnx_int64 k = 0; k < N; k++) {
        this->_TNs[k] =
          UniTensor(cytnx::random::normal({chi, vphys_dim[k], chi}, 0., 1.), false, 2);
        this->_TNs[k].set_labels({2 * k, 2 * k + 1, 2 * k + 2});
      }
    }

    void iMPS::_save_dispatch(fstream &f) {
      cytnx_uint64 N = this->_TNs.size();
      f.write((char *)&N, sizeof(cytnx_uint64));

      // save UniTensor one by one:
      for (cytnx_uint64 i = 0; i < N; i++) {
        this->_TNs[i]._Save(f);
      }
    }
    void iMPS::_load_dispatch(fstream &f) {
      cytnx_uint64 N;

      f.read((char *)&N, sizeof(cytnx_uint64));
      this->_TNs.resize(N);

      // Load UniTensor one by one:
      for (cytnx_uint64 i = 0; i < N; i++) {
        this->_TNs[i]._Load(f);
      }
    }

    Scalar iMPS::norm() const {
      cytnx_error_msg(true, "[Developing]%s", "\n");
      return Scalar(0);
    }

  }  // namespace tn_algo

}  // namespace cytnx
