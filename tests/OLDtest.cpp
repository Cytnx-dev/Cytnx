#include "cytnx.hpp"
#include <complex>
#include <cstdarg>
#include <functional>
#include <type_traits>
#include "hptt.h"
//#include "cutt.h"
using namespace std;
using namespace cytnx;

typedef cytnx::Accessor ac;

Scalar generic_func(const Tensor &input, const std::vector<Scalar> &args) {
  auto out = input + args[0];
  out = args[1] + out;
  return out(0, 0).item();
}

class test {
 public:
  Tensor tff(const Tensor &T) {
    auto A = T.reshape(2, 3, 4);
    return A;
  }
};

//-------------------------------------------

Tensor myfunc(const Tensor &Tin) {
  // Tin should be a 4x4 tensor.
  Tensor A = arange(25).reshape({5, 5});
  A += A.permute({1, 0}).contiguous() + 1;

  return linalg::Dot(A, Tin);
}

class MyOp2 : public LinOp {
 public:
  Tensor H;
  MyOp2() : LinOp("mv", 0, Type.Double, Device.cpu) {  // invoke base class constructor!
    auto T = arange(100).reshape(10, 10);
    T = T + T.permute(1, 0);
    print(linalg::Eigh(T));
  }

  UniTensor matvec(const UniTensor &in) {
    auto T = arange(100).reshape(10, 10);
    T = T + T.permute(1, 0);
    auto H = UniTensor(T, 1);

    auto out = Contract(H, in);
    out.set_labels(in.labels());

    return out;
  }
};

#define cuttCheck(stmt)                                                               \
  do {                                                                                \
    cuttResult err = stmt;                                                            \
    if (err != CUTT_SUCCESS) {                                                        \
      fprintf(stderr, "%s in file %s, function %s\n", #stmt, __FILE__, __FUNCTION__); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

Scalar run_DMRG(tn_algo::MPO &mpo, tn_algo::MPS &mps, int Nsweeps,
                std::vector<tn_algo::MPS> ortho_mps = {}, double weight = 40) {
  auto model = tn_algo::DMRG(mpo, mps, ortho_mps, weight);

  model.initialize();
  Scalar E;
  for (int xi = 0; xi < Nsweeps; xi++) {
    E = model.sweep();
    cout << "sweep " << xi << "/" << Nsweeps << " | Enr: " << E << endl;
  }
  return E;
}

int main(int argc, char *argv[]) {
  print(User_debug);

  Storage s112 = zeros(10).storage();
  s112.at<double>(4);

  // User_debug=false;

  print(User_debug);
  s112.at<float>(4);

  return 0;

  auto T1 = UniTensor(arange(30).reshape(2, 5, 3), 1);
  auto T2 = T1.clone().relabels({0, 3, 4});
  auto T3 = T1.clone().relabels({5, 3, 7});

  T1.print_diagram();
  T2.print_diagram();
  T3.print_diagram();

  auto Ott = Contracts(T1, T2, T3);

  Ott.print_diagram();
  return 0;

  // testing Sparse:
  auto bdi = Bond(4, BD_IN, {{0}, {-2}, {+2}, {0}});
  auto bdo = bdi.redirect();
  auto phys_bdi = Bond(2, BD_IN, {{1}, {-1}});
  auto phys_bdo = phys_bdi.redirect();

  auto U1 = UniTensor({bdi, bdo, phys_bdi, phys_bdo}, {}, 2);

  U1.print_diagram();
  print(U1);
  return 0;
  // I
  U1.at({0, 0, 0, 0}) = 1;
  U1.at({0, 0, 1, 1}) = 1;
  U1.at({3, 3, 0, 0}) = 1;
  U1.at({3, 3, 1, 1}) = 1;

  // S-
  U1.at({0, 1, 1, 0}) = 1;

  // S+
  U1.at({0, 2, 0, 1}) = 2;

  // S+
  U1.at({1, 3, 0, 1}) = 1;

  // S-
  U1.at({2, 3, 1, 0}) = 4;

  print(U1);

  U1.permute({1, 0, 3, 2});
  U1.set_rowrank(3);
  U1.contiguous_();

  print(U1);

  // U1.set_rowrank(3);
  U1.Save("sps");

  auto readU1 = UniTensor::Load("sps.cytnx");

  return 0;

  auto A0 = UniTensor({Bond(1, BD_KET, {{0}}), phys_bdi, phys_bdi, Bond(1, BD_BRA, {{0}})}, {}, 2);
  A0.get_block_(0).item() = 1;

  return 0;

  auto X0 = arange(32).reshape(2, 2, 2, 2, 2);

  auto X1 = X0.permute(0, 3, 2, 1, 4);
  auto X2 = X0.permute(2, 0, 1, 4, 3);

  auto X1c = X1.contiguous();
  auto X2c = X2.contiguous();

  cout << X1c + X2c << endl;
  cout << X1c.is_contiguous() << X2c.is_contiguous() << endl;

  X1c += X2c;

  X1 += X2;

  cout << X1 << endl;
  cout << X2 << endl;
  auto idd = vec_range(5);
  cout << X1._impl->invmapper() << endl;
  cout << X1._impl->mapper() << endl;

  cout << X2._impl->invmapper() << endl;
  cout << X2._impl->mapper() << endl;
  cout << vec_map(idd, X2._impl->invmapper()) << endl;

  return 0;

  print(X1._impl->mapper());
  print(X1._impl->invmapper());

  print(X2._impl->mapper());
  print(X2._impl->invmapper());

  std::vector<cytnx_uint32> _invL2R(7);

  for (int i = 0; i < X1._impl->mapper().size(); i++) {
    _invL2R[i] = X2._impl->mapper()[X1._impl->invmapper()[i]];
  }
  print(_invL2R);

  return 0;

  return 0;

  auto Tc = zeros({4, 1, 1});
  Tc.Conj_();

  auto L0 = UniTensor(zeros({4, 1, 1}), 0);  //#Left boundary
  auto R0 = UniTensor(zeros({4, 1, 1}), 0);  //#Right boundary
  L0.get_block_()(0, 0, 0) = 1.;
  R0.get_block_()(3, 0, 0) = 1.;

  MyOp2 OP;
  auto v = UniTensor(random::normal({10}, 0, 1, -1, 99), 1);

  // auto out = OP.matvec(v);

  print(linalg::Lanczos_Gnd_Ut(&OP, v));

  return 0;

  auto ac1 = Accessor::qns({{1}, {-1}});
  print(ac1);

  auto tn1 = zeros(4);
  auto tn3 = zeros(7);

  cout << algo::Concatenate(tn1, tn3);

  return 0;

  return 0;
}
