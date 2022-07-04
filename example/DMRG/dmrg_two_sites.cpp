#include "cytnx.hpp"
#include <cfloat>
#define min(a, b) (a < b ? a : b)

using namespace cytnx;

/*
Reference: https://www.tensors.net
Author: j9263178
*/

class Hxx : public LinOp {
 public:
  Network projector;
  UniTensor L, M1, M2, R;
  Hxx(Network& projector, UniTensor& L, UniTensor& M1, UniTensor& M2, UniTensor& R, int dim)
      : LinOp("mv", dim, Type.Double, Device.cpu) {
    this->projector = projector;
    this->L = L;
    this->M1 = M1;
    this->M2 = M2;
    this->R = R;
  }

  // Overload matvec for custom operation:
  Tensor matvec(const Tensor& psi) override {
    Tensor psic = psi.clone();
    psic.reshape_((this->L).shape()[1], (this->M1).shape()[2], (this->M2).shape()[2],
                  (this->R).shape()[1]);
    projector.PutUniTensor("psi", UniTensor(psic, 0), false);
    auto out = projector.Launch(true).get_block_();
    out.flatten_();
    return out;
  }
};

std::vector<UniTensor> DMRG(std::vector<UniTensor>& A, UniTensor& ML, UniTensor& M, UniTensor& MR,
                            int chi, int Nsweeps, int maxit, int krydim) {
  std::vector<UniTensor> out, svdtemp;
  UniTensor s, u, vT;
  int chid = M.shape()[2];
  int Nsites = A.size();
  int chil, chir;

  Network L_AMAH, R_AMAH, projector;
  projector.Fromfile("projector.net");
  L_AMAH.Fromfile("L_AMAH.net");
  R_AMAH.Fromfile("R_AMAH.net");

  std::vector<UniTensor> LR(Nsites + 1);
  LR[0] = ML;
  LR[Nsites] = MR;

  // Setup : put MPS into right/left? othogonal form
  for (int p = 0; p < Nsites - 1; p++) {
    // SVD on A[p]
    svdtemp = linalg::Svd(A[p]);
    s = svdtemp[0];
    u = svdtemp[1];
    vT = svdtemp[2];

    // A[p+1] absorbs s and vT from A[p]
    A[p] = u;
    A[p + 1] = Contract(Contract(s, vT), A[p + 1]);

    // Calculate and store all the Ls for the right-to-left sweep

    L_AMAH.PutUniTensors({"L", "A", "A_Conj", "M"}, {LR[p], A[p], A[p].Conj(), M}, false);
    LR[p + 1] = L_AMAH.Launch(true);
  }

  // SVD for the right boundary tensor A[Nsites-1], only save U (?)
  A[Nsites - 1] = linalg::Svd(A[Nsites - 1], true, false)[1];

  std::vector<cytnx_double> Ekeep(0);

  for (int k = 1; k < Nsweeps + 2; k++) {
    // Optimization sweep: right-to-left
    printf("\n L <- R \n");
    for (int p = Nsites - 2; p > -1; p--) {
      // A[p] is absorbed to make a two-site update
      chil = A[p].shape()[0];
      chir = A[p + 1].shape()[2];
      // std::cout<<"R -> L A[p] : "<<A[p].labels()<< "\nA[p+1] : "<<A[p+1].labels()<<"p =
      // "<<p<<std::endl;
      auto psi = Contract(A[p], A[p + 1]);
      auto psilabel = psi.labels();
      auto psi_T = psi.get_block_();
      psi_T.flatten_();

      //[ERROR] [M1] and [21948056] has same_data.
      // projector.PutUniTensors({"L", "M1", "M2", "R"}, {LR[p], M, M, LR[p+2]}, false);

      projector.PutUniTensor("M1", M);
      projector.PutUniTensors({"L", "M2", "R"}, {LR[p], M, LR[p + 2]}, false);

      auto H = Hxx(projector, LR[p], M, M, LR[p + 2], psi.shape()[0]);

      auto res = linalg::Lanczos_ER(&H, 1, true, maxit, DBL_MAX, false, psi_T, krydim, false);
      // auto res = linalg::Lanczos_Gnd(&H, 1.0e-14, true, psi_T, false, maxit);

      psi_T = res[1].reshape(chil, chid, chid, chir);
      psi = UniTensor(psi_T, 2);
      psi.set_labels(psilabel);
      Ekeep.push_back(Scalar(res[0].item()));

      // restore MPS via SVD
      int newdim = min(min(chil * chid, chir * chid), chi);
      svdtemp = linalg::Svd_truncate(psi, newdim);
      s = svdtemp[0];
      s.Div_(s.get_block_().Norm().item());
      u = svdtemp[1];
      vT = svdtemp[2];
      A[p] = Contract(u, s);
      A[p + 1] = vT;

      // get the new block Hamiltonian
      R_AMAH.PutUniTensors({"R", "B", "M", "B_Conj"}, {LR[p + 2], A[p + 1], M, A[p + 1].Conj()},
                           false);
      LR[p + 1] = R_AMAH.Launch(true);

      printf("Sweep: %d of %d, Loc: %d, Energy: ", k, Nsweeps, p);
      std::cout << Ekeep[Ekeep.size() - 1] << std::endl;

    }  // end of sweep for

    // SVD for the left boundary tensor, only save vT (?)
    A[0].set_rowrank(1);
    A[0] = linalg::Svd(A[0], false, true)[1];  // shape[1,2,2], rowrank = 1

    // Optimization sweep: left-to-right
    printf("\n L -> R \n");
    for (int p = 0; p < Nsites - 1; p++) {
      // std::cout<<"L -> R A[p] : "<<A[p].labels()<< "\nA[p+1] : "<<A[p+1].labels()<<"p =
      // \n\n"<<p<<std::endl;
      chil = A[p].shape()[0];
      chir = A[p + 1].shape()[2];
      auto psi = Contract(A[p], A[p + 1]);
      auto psilabel = psi.labels();
      auto psi_T = psi.get_block_();
      psi_T.flatten_();

      projector.PutUniTensor("M1", M);
      projector.PutUniTensors({"L", "M2", "R"}, {LR[p], M, LR[p + 2]}, false);

      auto H = Hxx(projector, LR[p], M, M, LR[p + 2], psi.shape()[0]);

      auto res = linalg::Lanczos_ER(&H, 1, true, maxit, DBL_MAX, false, psi_T, krydim, false);
      // auto res = linalg::Lanczos_Gnd(&H, 1.0e-14, true, psi_T, false, maxit);

      psi_T = res[1].reshape(chil, chid, chid, chir);
      psi = UniTensor(psi_T, 2);
      psi.set_labels(psilabel);
      Ekeep.push_back(Scalar(res[0].item()));

      // restore MPS via SVD

      int newdim = min(min(chil * chid, chir * chid), chi);
      svdtemp = linalg::Svd_truncate(psi, newdim);

      // /home/j9263178/cytnx_test/lsq.cpp:98: 未定義參考到 cytnx::UniTensor
      // cytnx::operator/<cytnx::Scalar::Sproxy>(cytnx::UniTensor const&, cytnx::Scalar::Sproxy
      // const&) /home/j9263178/cytnx_test/lsq.cpp:136: 未定義參考到 cytnx::UniTensor
      // cytnx::operator/<cytnx::Scalar::Sproxy>(cytnx::UniTensor const&, cytnx::Scalar::Sproxy
      // const&) collect2: 錯誤：ld 回傳 1
      s = svdtemp[0];
      s.Div_(s.get_block_().Norm().item());
      u = svdtemp[1];
      vT = svdtemp[2];
      A[p] = u;
      A[p + 1] = Contract(s, vT);

      L_AMAH.PutUniTensors({"L", "A", "A_Conj", "M"}, {LR[p], A[p], A[p].Conj(), M}, false);
      LR[p + 1] = L_AMAH.Launch(true);

      printf("Sweep: %d of %d, Loc: %d, Energy: ", k, Nsweeps, p);
      std::cout << Ekeep[Ekeep.size() - 1] << std::endl;

    }  // end of iteration for

    // SVD for the right boundary tensor, only save U (?)
    A[Nsites - 1].set_rowrank(2);
    A[Nsites - 1] = linalg::Svd(A[Nsites - 1], true, false)[1];  // shape[1,2,2], rowrank = 2

  }  // end of iteration for

  return out;
}

int main() {
  int Nsites = 20;  // system size
  int chid = 2;  // physical local dimension
  int chi = 32;  // bond dimension
  int Nsweeps = 4;  // number of DMRG sweeps
  int krydim = 4;  // dimension of Krylov subspace
  int maxit = 4;  // iterations of Lanczos method

  // ?
  // std::complex<double> j = (0, 1);
  // auto Sx = physics::spin(0.5,'x');
  // auto Sy = physics::spin(0.5,'y');
  // auto Sp = Sx + j*Sy;
  // auto Sm = Sx - j*Sy;

  auto Sp = zeros({2, 2});
  Sp.at<cytnx_double>(0, 1) = 1;
  auto Sm = zeros({2, 2});
  Sm.at<cytnx_double>(1, 0) = 1;

  auto Si = eye(2);

  auto M = zeros({4, 4, chid, chid}, Type.Double);
  M(0, 0, ":", ":") = Si;
  M(0, 1, ":", ":") = sqrt(2) * Sm;
  M(0, 2, ":", ":") = sqrt(2) * Sp;
  M(1, 3, ":", ":") = sqrt(2) * Sp;
  M(2, 3, ":", ":") = sqrt(2) * Sm;
  M(3, 3, ":", ":") = Si;

  // std::cout<<M<<std::endl;

  auto M_ = UniTensor(M, 0);

  auto ML = UniTensor(zeros({4, 1, 1}, Type.Double), 0);  // left MPO boundary
  auto MR = UniTensor(zeros({4, 1, 1}, Type.Double), 0);  // right MPO boundary
  ML.get_block_()(0, 0, 0) = 1;
  MR.get_block_()(3, 0, 0) = 1;

  std::vector<UniTensor> A(Nsites);
  A[0] = UniTensor(zeros({1, chid, min(chi, chid)}), 2);
  random::Make_normal(A[0].get_block_(), 0, 1);

  for (int k = 1; k < Nsites; k++) {
    int pre = A[k - 1].shape()[2];
    int nxt = min(min(chi, A[k - 1].shape()[2] * chid), pow(chid, (Nsites - k - 1)));
    A[k] = UniTensor(zeros({pre, chid, nxt}), 2);
    random::Make_normal(A[k].get_block_(), 0, 1);
    A[k].set_labels({2 * k, 2 * k + 1, 2 * k + 2});
  }

  DMRG(A, ML, M_, MR, chi, Nsweeps, maxit, krydim);

  printf("\ndone\n");
}