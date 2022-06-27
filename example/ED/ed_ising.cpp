#include "cytnx.hpp"
#include <iostream>

using namespace std;
using namespace cytnx;
namespace cy = cytnx;
typedef Accessor ac;

class Hising : public cy::LinOp {
 public:
  cytnx_double J, Hx;
  cytnx_uint32 L;

  Hising(cytnx_uint32 L, cytnx_double J, cytnx_double Hx)
      : cy::LinOp("mv", pow(2, L), Type.Double,
                  Device.cpu)  // rememeber to invoke base class constructor
  {
    // custom members
    this->J = J;
    this->Hx = Hx;
    this->L = L;
  }

  double SzSz(const cytnx_uint32 &i, const cytnx_uint32 &j, const cytnx_uint32 &ipt_id,
              cytnx_uint32 &out_id) {
    out_id = ipt_id;
    return (1. - 2. * (((ipt_id >> i) & 0x1) ^ ((ipt_id >> j) & 0x1)));
  }
  double Sx(const cytnx_uint32 &i, const cytnx_uint32 &ipt_id, cytnx_uint32 &out_id) {
    out_id = ipt_id ^ ((0x1) << i);
    return 1.0;
  }

  // let's overload this with custom operation:
  Tensor matvec(const Tensor &v) {
    auto out = zeros(v.shape()[0], v.dtype(), v.device());
    cytnx_uint32 oid;
    double amp;

    for (cytnx_uint32 a = 0; a < v.shape()[0]; a++) {
      for (cytnx_uint32 i = 0; i < this->L; i++) {
        amp = this->SzSz(i, (i + 1) % this->L, a, oid);
        out[{ac(oid)}] += amp * this->J * v[{ac(a)}];

        amp = this->Sx(i, a, oid);
        out[{ac(oid)}] += amp * (-this->Hx) * v[{ac(a)}];
      }
    }
    return out;
  }
};

int main(int argc, char *argv[]) {
  cytnx_uint32 L = 4;
  double J = 1;
  double Hx = 0.3;
  auto H = Hising(L, J, Hx);
  cout << cy::linalg::Lanczos_ER(&H, 3) << endl;
  return 0;
}
