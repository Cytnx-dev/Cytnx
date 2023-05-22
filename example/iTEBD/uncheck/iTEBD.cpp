#include "cytnx.hpp"
#include <iostream>
#include <cmath>  // abs

using namespace std;
using namespace cytnx;

typedef cytnx::Accessor ac;

int main(int argc, char *argv[]) {
  unsigned int chi = 20;
  double J = 1.0;
  double Hx = 1.0;
  double CvgCrit = 1.0e-10;
  double dt = 0.1;

  //> Create onsite-Op
  Tensor Sz = cytnx::zeros(2*2).reshape({2,2});
  Sz.at<double>({0, 0}) = 1;
  Sz.at<double>({1, 1}) = -1;

  Tensor Sx = cytnx::zeros(2*2).reshape({2,2});
  Sx.at<double>({0, 1}) = Sx.at<double>({1, 0}) = Hx;
  Tensor I = Sz.clone();
  I.at<double>({1, 1}) = 1;

  // cout << Sz << Sx << I << endl;

  //> Build Evolution Operator
  Tensor TFterm = cytnx::linalg::Kron(Sx, I) + cytnx::linalg::Kron(I, Sx);
  Tensor ZZterm = cytnx::linalg::Kron(Sz, Sz);
  Tensor tH = Hx * TFterm + J * ZZterm;

  cout<<"tH::::::"<<tH<<endl;

  Tensor teH = cytnx::linalg::ExpH(tH, -dt);
  teH.reshape_({2, 2, 2, 2});
  cout << teH;
  tH.reshape_({2, 2, 2, 2});

  UniTensor eH = UniTensor(teH, false, 2);
  eH.print_diagram();
  cout << eH;

  UniTensor H = UniTensor(tH, false, 2);
  H.print_diagram();

  //> Create MPS:
  //
  //     |    |
  //   --A-la-B-lb--
  //
  UniTensor A = UniTensor({cytnx::Bond(chi), cytnx::Bond(2), cytnx::Bond(chi)}, vector<string>({"-1", "0", "-2"}), 1);
  UniTensor B = UniTensor(A.bonds(), vector<string>({"-3", "1", "-4"}), 1);

  cytnx::random::Make_normal(B.get_block_(), 0, 0.2);
  cytnx::random::Make_normal(A.get_block_(), 0, 0.2);

  UniTensor la =
    UniTensor({cytnx::Bond(chi), cytnx::Bond(chi)}, vector<string>({"-2", "-3"}), 1, Type.Double, Device.cpu, true);
  UniTensor lb =
    UniTensor({cytnx::Bond(chi), cytnx::Bond(chi)}, vector<string>({"-4", "-5"}), 1, Type.Double, Device.cpu, true);
  la.put_block(cytnx::ones(chi));
  lb.put_block(cytnx::ones(chi));
  //> Evov:
  double Elast = 0;

  for (unsigned int i = 0; i < 10000; i++) {
    A.set_labels({"-1", "0", "-2"});
    B.set_labels({"-3", "1", "-4"});
    la.set_labels({"-2", "-3"});
    lb.set_labels({"-4", "-5"});

    // contract all
    UniTensor X = cytnx::Contract(cytnx::Contract(A, la), cytnx::Contract(B, lb));
    lb.set_label(lb.get_index("-5"), "-1");
    X = cytnx::Contract(lb, X);

    // X =
    //           (0)  (1)
    //            |    |
    //  (-4) --lb-A-la-B-lb-- (-5)
    //
    UniTensor Xt = X.clone();

    //> calculate norm and energy for this step
    // Note that X,Xt contract will result a rank-0 tensor, which can use item() toget element
    double XNorm = cytnx::Contract(X, Xt).item<double>();
    UniTensor XH = cytnx::Contract(X, H);

    // X.print_diagram();
    // H.print_diagram();
    // XH.print_diagram();

    XH.set_labels({"-4", "-5", "0", "1"});
    double XHX = cytnx::Contract(Xt, XH).item<double>();
    double E = XHX / XNorm;

    //> check if converged.
    if (abs(E - Elast) < CvgCrit) {
      cout << "[Converged!]" << endl;
      break;
    }
    cout << "Step: " << i << " Enr: " << Elast << endl;
    Elast = E;

    //> Time evolution the MPS
    UniTensor XeH = cytnx::Contract(X, eH);
    XeH.permute_({"-4", "2", "3", "-5"});

    //> Do Svd + truncate
    //
    //        (2)   (3)                   (2)                                    (3)
    //         |     |          =>         |         +   (-6)--s--(-7)  +         |
    //  (-4) --= XeH =-- (-5)        (-4)--U--(-6)                          (-7)--Vt--(-5)
    //
    XeH.set_rowrank(2);
    vector<UniTensor> out = cytnx::linalg::Svd_truncate(XeH, chi);
    la = out[0];
    A = out[1];
    B = out[2];
    double Norm = cytnx::linalg::Norm(la.get_block_()).item<double>();
    la *= 1. / Norm;  // normalize

    // de-contract the lb tensor , so it returns to
    //
    //            |     |
    //       --lb-A'-la-B'-lb--
    //
    // again, but A' and B' are updated
    A.set_labels({"-1", "0", "-2"});
    A.set_rowrank(1);
    B.set_labels({"-3", "1", "-4"});
    B.set_rowrank(1);

    UniTensor lb_inv = 1. / lb;
    A = cytnx::Contract(lb_inv, A);
    B = cytnx::Contract(B, lb_inv);

    //> translation symm, exchange A and B site
    UniTensor tmp = A;
    A = B;
    B = tmp;

    tmp = la;
    la = lb;
    lb = tmp;
  }

  return 0;
}
