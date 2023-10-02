#include <iostream>
#include "cytnx.hpp"

using namespace std;
using namespace cytnx;

int main(int argc, char* argv[]) {
  std::vector<long> v = {2, 3};
  Storage sd = Storage::from_vector(v);

  cout << sd.dtype() << endl;

  auto fille = float(10);
  sd.fill(fille);

  std::cout << sd << std::endl;
  return 0;

  static std::vector<unsigned int> dtype_list = {
    // Type.Void,
    Type.ComplexDouble, Type.ComplexFloat, Type.Double, Type.Float, Type.Int64,
    Type.Uint64,        Type.Int32,        Type.Uint32, Type.Int16, Type.Uint16};
  for (auto dtype : dtype_list) {
    std::vector<Tensor> Ts = {Tensor({3, 4}, dtype).to(cytnx::Device.cuda),
                              Tensor({2, 4}, dtype).to(cytnx::Device.cuda),
                              Tensor({5, 4}, dtype).to(cytnx::Device.cuda)};
    // InitTensorUniform(Ts);
    Tensor vstack_tens = algo::Vstack(Ts);
  }
  return 0;
}
