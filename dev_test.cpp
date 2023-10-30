#include <iostream>
#include "cytnx.hpp"

//#include <torch/torch.h>

using namespace std;
using namespace cytnx;

int main(int argc, char* argv[]) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  bool is_diag = true;
  auto ut_diag = UniTensor({Bond(4), Bond(4)}, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut_diag, -100.0, 100.0, seed);
  std::vector<cytnx_int64> dst_shape = {4, 4};
  auto dst_ut1 = ut_diag.reshape(dst_shape, 1);
  auto dst_ut2 = ut_diag.clone();

  auto ut_tr = dst_ut2.Trace(0, 1);

  auto ut_dense = dst_ut2.to_dense();
  auto ans = ut_dense.Trace(0, 1);

  print(ut_tr);
  print(ans);
  // print(dst_ut2.is_diag());
  // print(dst_ut2.shape());
  // dst_ut2.reshape_(dst_shape, 1);

  // print(dst_ut2.shape());

  return 0;

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
