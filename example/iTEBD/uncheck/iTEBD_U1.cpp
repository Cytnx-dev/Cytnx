#include "cytnx.hpp"
#include <iostream>
#include <cmath>  // abs

using namespace std;
using namespace cytnx;

////
// Author: Kai-Hsin Wu
////

int main(int argc, char *argv[]) {
  // Example of 1D Heisenberg model
  //// iTEBD
  ////-------------------------------------

  cytnx_int64 chi = 40;
  cytnx_double J = 1.0;
  cytnx_double CvgCrit = 1.0e-12;
  cytnx_double dt = 0.1;

  //// Create Si Sj local H with symmetry:
  //// SzSz + S+S- + h.c.
  Bond bdi = Bond(BD_KET, {{1}, {-1}}, {1, 1});
  Bond bdo = bdi.clone().set_type(BD_BRA);
  UniTensor H = UniTensor({bdi, bdi, bdo, bdo}, vector<cytnx_int64>({2, 3, 1, 0}));
  //  H = UniTensor({bdi,bdi,bdo,bdo},labels={2,3,0,1});

  //// assign:
  // Q = 2  // Q = 0:    // Q = -2:
  // {1}    {{ -1, 1}     {1}
  //         {  1,-1}}
  // H.print_blocks();
  H.get_block_({0, 0, 0, 0}).at({0, 0, 0, 0}) = 1;
  // Tensor T0 = H.get_block_({1,1});
  H.get_block_({0, 1, 0, 1}).at({0, 0, 0, 0}) = -1;
  H.get_block_({1, 0, 1, 0}).at({0, 0, 0, 0}) = -1;
  H.get_block_({0, 1, 1, 0}).at({0, 0, 0, 0}) = 1;
  H.get_block_({1, 0, 0, 1}).at({0, 0, 0, 0}) = 1;
  // T0.at({0,0}) = -1,T0.at({1,1}) = -1;
  // T0.at({0,1}) = T0.at({1,0}) = 1;
  H.get_block_({1, 1, 1, 1}).at({0, 0, 0, 0}) = 1;
  // H.get_block_({2,2}).at({0,0}) = 1;

  // //// create gate:
  UniTensor eH = linalg::ExpH(H, -dt);

  // //// Create MPS:
  //
  //     |    |
  //   --A-la-B-lb--
  //
  Bond bd_mid = bdi.combineBond(bdi);
  UniTensor A = UniTensor({bdi, bdi, bd_mid.redirect()}, vector<string>({"-1", "0", "-2"}));
  UniTensor B = UniTensor({bd_mid, bdi, bdo}, vector<string>({"-3", "1", "-4"}));

  for (int b = 0; b < B.get_blocks_().size(); b++) {
    random::Make_normal(B.get_block_(b), 0, 0.2);
  }
  for (int a = 0; a < A.get_blocks_().size(); a++) {
    random::Make_normal(A.get_block_(a), 0, 0.2);
  }
  A.print_diagram();
  B.print_diagram();

  UniTensor la = UniTensor({bd_mid, bd_mid.redirect()}, vector<string>({"-2", "-3"}), -1,
                           Type.Double, Device.cpu, true);
  UniTensor lb =
    UniTensor({bdi, bdo}, vector<string>({"-4", "-5"}), -1, Type.Double, Device.cpu, true);

  for (int b = 0; b < lb.get_blocks_().size(); b++) {
    lb.get_block_(b).fill(1);
  }

  for (int a = 0; a < la.get_blocks_().size(); a++) {
    la.get_block_(a).fill(1);
  }
  la.print_diagram();
  lb.print_diagram();

  //// Evov:
  cytnx_double Elast = 0;
  for (int i = 0; i < 10000; i++) {
    A.set_labels({"-1", "0", "-2"});
    B.set_labels({"-3", "1", "-4"});
    la.set_labels({"-2", "-3"});
    lb.set_labels({"-4", "-5"});

    //// contract all
    UniTensor X = Contract(Contract(A, la), Contract(B, lb));
    lb.set_label(lb.get_index("-5"), "-1");
    X = Contract(lb, X);

    //// X =
    //           (0)  (1)
    //            |    |
    //  (-4) --lb-A-la-B-lb-- (-5)
    //
    //// calculate local energy:
    //// <psi|psi>
    UniTensor Xt = X.Dagger();
    cytnx_double XNorm = double(Contract(X, Xt).item().real());

    //// <psi|H|psi>
    UniTensor XH = Contract(X, H);
    XH.set_labels({"-4", "-5", "0", "1"});
    cytnx_double XHX = double(Contract(Xt, XH).item().real());

    cytnx_double E = XHX / XNorm;

    //// check if converged.
    if (abs(E - Elast) < CvgCrit) {
      cout << "{Converged!}" << endl;
      break;
    }
    printf("Step: %d Enr: %5.8f\n", i, Elast);
    Elast = E;

    //// Time evolution the MPS
    UniTensor XeH = Contract(X, eH);
    XeH.permute_({"-4", "2", "3", "-5"}, -1, true);

    //// Do Svd + truncate
    ////
    //        (2)   (3)                   (2)                                    (3)
    //         |     |          =>         |         +   (-6)--s--(-7)  +         |
    //  (-4) --= XeH =-- (-5)        (-4)--U--(-6)                          (-7)--Vt--(-5)
    //

    XeH.set_rowrank(2);
    if (XeH.shape()[0] * XeH.shape()[1] > chi) {
      auto tmp = linalg::Svd_truncate(XeH, chi);
      la = tmp[0];
      A = tmp[1];
      B = tmp[2];
    } else {
      auto tmp = linalg::Svd(XeH);
      la = tmp[0];
      A = tmp[1];
      B = tmp[2];
    }
    cytnx_double Norm = 0;
    for (int a = 0; a < la.get_blocks_().size(); a++) {
      Norm += double(
        (linalg::Norm(la.get_block_(a)).item() * linalg::Norm(la.get_block_(a)).item()).real());
    }
    Norm = sqrt(Norm);
    for (int a = 0; a < la.get_blocks_().size(); a++) {
      Tensor T = la.get_block_(a);
      T /= Norm;
    }
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

    UniTensor lb_inv = lb.clone();
    for (int b = 0; b < lb_inv.get_blocks_().size(); b++) {
      Tensor T = 1.0 / lb_inv.get_block_(b);
      lb_inv.put_block_(T, b);
    }

    A = Contract(lb_inv, A);
    B = Contract(B, lb_inv);

    // translation symmetry, exchange A and B site
    // A,B = B,A
    swap(A, B);
    swap(la, lb);
    // la,lb = lb,la
  }
}
