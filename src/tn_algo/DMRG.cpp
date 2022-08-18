#include "tn_algo/DMRG.hpp"
#include "Generator.hpp"
#include "Network.hpp"
#include "LinOp.hpp"
#include "linalg.hpp"
#include <tuple>
#include <iomanip>
#include <iostream>
#include "utils/vec_print.hpp"
using namespace std;

namespace cytnx {
  namespace tn_algo {

    //----------------------------
    // Internal function calls and objects!:

    class Hxx_new : public LinOp {
     public:
      Network anet;
      // std::vector<cytnx_int64> shapes;
      std::vector<UniTensor> ortho_mps;
      double weight;
      int counter;
      Hxx_new(std::vector<UniTensor> functArgs, const std::vector<UniTensor> &ortho_mps,
              const double &weight, const cytnx_int64 &dtype, const cytnx_int64 &device)
          : LinOp("mv", 0 /*doesn't matter for UniTensor as ipt*/, dtype, device) {
        UniTensor &L = functArgs[0];
        UniTensor &M1 = functArgs[1];
        UniTensor &M2 = functArgs[2];
        UniTensor &R = functArgs[3];

        // std::vector<cytnx_int64> pshape =
        // {L.shape()[1],M1.shape()[2],M2.shape()[2],R.shape()[1]}; vec_print(std::cout,pshape);
        this->anet.FromString({"psi: -1,-2;-3,-4", "L: ;-5,-1,0", "R: ;-7,-4,3", "M1: ;-5,-6,-2,1",
                               "M2: ;-6,-7,-3,2", "TOUT: 0,1;2,3"});
        this->anet.PutUniTensor("M2", M2);
        this->anet.PutUniTensors({"L", "M1", "R"}, {L, M1, R});

        this->ortho_mps = ortho_mps;
        this->weight = weight;
        this->counter = 0;
      }

      UniTensor matvec(const UniTensor &v) override {
        auto lbls = v.labels();

        this->anet.PutUniTensor("psi", v);
        UniTensor out = this->anet.Launch(true);  // get_block_ without copy

        // shifted ortho state:
        for (cytnx_int64 ir = 0; ir < this->ortho_mps.size(); ir++) {
          auto r = this->ortho_mps[ir].relabels(v.labels());
          Scalar c = Contract(r.Dagger(), v).item();
          out += this->weight * c * r;
        }
        out.set_labels(lbls);

        return out.contiguous();
      }
    };

    class Hxx : public LinOp {
     public:
      Network anet;
      std::vector<cytnx_int64> shapes;
      std::vector<Tensor> ortho_mps;
      double weight;
      int counter;

      Hxx(const cytnx_uint64 &psidim, std::vector<UniTensor> functArgs,
          const std::vector<Tensor> &ortho_mps, const double &weight, const cytnx_int64 &dtype,
          const cytnx_int64 &device)
          : LinOp("mv", psidim, dtype, device) {
        UniTensor &L = functArgs[0];
        UniTensor &M1 = functArgs[1];
        UniTensor &M2 = functArgs[2];
        UniTensor &R = functArgs[3];

        std::vector<cytnx_int64> pshape = vec_cast<cytnx_uint64, cytnx_int64>(
          {L.shape()[1], M1.shape()[2], M2.shape()[2], R.shape()[1]});
        // vec_print(std::cout,pshape);
        this->anet.FromString({"psi: ;-1,-2,-3,-4", "L: ;-5,-1,0", "R: ;-7,-4,3", "M1: ;-5,-6,-2,1",
                               "M2: ;-6,-7,-3,2", "TOUT: ;0,1,2,3"});
        this->anet.PutUniTensor("M2", M2);
        this->anet.PutUniTensors({"L", "M1", "R"}, {L, M1, R});

        this->shapes = pshape;
        this->ortho_mps = ortho_mps;
        this->weight = weight;
        this->counter = 0;
      }

      Tensor matvec(const Tensor &v) override {
        auto v_ = v.clone();

        auto psi_u = UniTensor(v_, false, 0);  // ## share memory, no copy
        psi_u.reshape_(this->shapes);
        this->anet.PutUniTensor("psi", psi_u);
        Tensor out = this->anet.Launch(true).get_block_();  // get_block_ without copy
        out.flatten_();  // only change meta, without copy.

        // shifted ortho state:

        for (cytnx_int64 ir = 0; ir < this->ortho_mps.size(); ir++) {
          auto r = this->ortho_mps[ir];
          auto c = linalg::Dot(r, v).item();
          out += this->weight * c * r;
        }
        // cout << counter << endl; counter++;
        // cout << out << endl;
        return out;
      }
    };

    std::vector<Tensor> optimize_psi(Tensor psivec, std::vector<UniTensor> functArgs,
                                     const cytnx_uint64 &maxit = 4000,
                                     const cytnx_uint64 &krydim = 4,
                                     std::vector<Tensor> ortho_mps = {},
                                     const double &weight = 30) {
      auto H =
        Hxx(psivec.shape()[0], functArgs, ortho_mps, weight, psivec.dtype(), psivec.device());

      auto out = linalg::Lanczos_Gnd(&H, 1.0e-12, true, psivec, false, maxit);
      return out;
    }

    std::vector<UniTensor> optimize_psi_new(UniTensor psivec, std::vector<UniTensor> functArgs,
                                            const cytnx_uint64 &maxit = 4000,
                                            const cytnx_uint64 &krydim = 4,
                                            std::vector<UniTensor> ortho_mps = {},
                                            const double &weight = 30) {
      auto H = Hxx_new(functArgs, ortho_mps, weight, psivec.dtype(), psivec.device());

      auto out = linalg::Lanczos_Gnd_Ut(&H, psivec, 1.0e-12, true, false, maxit);

      return out;
    }

    //----------------------------

    void DMRG_impl::initialize() {
      // initialize everything
      // 1. setting env:

      // Initialiaze enviroment:
      auto L0 =
        UniTensor(cytnx::zeros({this->mpo.get_op(0).shape()[0], 1, 1}), false, 0);  // Left boundary
      auto R0 = UniTensor(cytnx::zeros({this->mpo.get_op(this->mps.size() - 1).shape()[1], 1, 1}),
                          false, 0);  // Right boundary
      L0.get_block_()(0, 0, 0) = 1.;
      R0.get_block_()(-1, 0, 0) = 1.;

      // Put in the left normalization form and calculate transfer matrices LR
      /*
       LR[0]:        LR[1]:            LR[2]:

          -----      -----A[0]---     -----A[1]---
          |          |     |          |     |
         ML----     LR[0]--M-----    LR[1]--M-----      ......
          |          |     |          |     |
          -----      -----A*[0]--     -----A*[1]--


        L_AMAH.net file is used to contract the LR[i]
        >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      */
      this->LR.resize(this->mps.size() + 1);
      this->LR[0] = L0;
      this->LR.back() = R0;
      this->mps.Into_Lortho();

      for (int p = 0; p < this->mps.size() - 1; p++) {
        // this->mps.S_mvright();
        // anet = cytnx.Network("L_AMAH.net")
        // anet.PutUniTensors(["L","A","A_Conj","M"],[self.LR[p],self.mps.A[p],self.mps.A[p].Conj(),self.mpo.get_op(p)],is_clone=False);

        // hard coded the network:
        auto Lenv = this->LR[p].relabels({-2, -1, -3});
        auto tA = this->mps.data()[p].relabels({-1, -4, 1});
        auto tAc = this->mps.data()[p].Conj();
        tAc.set_labels({-3, -5, 2});
        auto M = this->mpo.get_op(p).relabels({-2, 0, -4, -5});
        this->LR[p + 1] = Network::Contract({Lenv, tA, tAc, M}, ";0,1,2").Launch(true);
      }
      // this->mps.S_mvright();

      // prepare if calculate excited states:
      this->hLRs.resize(this->ortho_mps.size());
      for (cytnx_int64 ip = 0; ip < this->ortho_mps.size(); ip++) {
        auto omps = this->ortho_mps[ip];

        // init environ:
        auto hL0 = UniTensor(zeros({1, 1}), false, 0);  // Left boundary
        auto hR0 = UniTensor(zeros({1, 1}), false, 0);  // Right boundary
        hL0.get_block_()(0, 0) = 1.;
        hR0.get_block_()(0, 0) = 1.;

        this->hLRs[ip].resize(this->mps.size() + 1);

        // hLR is the alias/ref:
        auto &hLR = hLRs[ip];

        hLR[0] = hL0;
        hLR.back() = hR0;

        // anet = cytnx.Network("hL_AMAH.net")
        auto anet = Network();
        anet.FromString({"hL: ;-1,-2", "Av: -1,-4;1", "Ap: -2,-4;2", "TOUT: ;1,2"});
        for (cytnx_int64 p = 0; p < this->mps.size() - 1; p++) {
          // anet.PutUniTensors(["hL","Av","Ap"],[hLR[p],self.mps.A[p],omps.A[p].Conj()],is_clone=False);

          anet.PutUniTensors({"hL", "Av", "Ap"},
                             {hLR[p], this->mps.data()[p], omps.data()[p].Conj()});
          hLR[p + 1] = anet.Launch(true);
        }
      }

    }  // DMRG_impl::initialize

    Scalar DMRG_impl::sweep(const bool &verbose, const cytnx_int64 &maxit,
                            const cytnx_int64 &krydim) {
      Scalar Entemp;

      // a. Optimize from right-to-left:
      /*
        psi:                   Projector:

          --A[p]--A[p+1]--s--              --         --
             |       |                     |    | |    |
                                          LR[p]-M-M-LR[p+1]
                                           |    | |    |
                                           --         --
        b. Transfer matrix from right to left :
         LR[-1]:       LR[-2]:

             ---          ---A[-1]---
               |               |    |
             --MR         -----M--LR[-1]   ......
               |               |    |
             ---          ---A*[-1]--

        c. For Right to Left, we want A's to be in shape
                   -------------
                  /             \
         virt ____| chi     chi |____ virt
                  |             |
         phys ____| 2           |
                  \             /
                   -------------
      */

      for (cytnx_int64 p = this->mps.size() - 2; p > -1;
           p--) {  // in range(self.mps.Nsites()-2,-1,-1):
        if (verbose) std::cout << "  [<<] upd loc:" << p << "| " << std::flush;

        cytnx_int64 dim_l = this->mps.data()[p].shape()[0];
        cytnx_int64 dim_r = this->mps.data()[p + 1].shape()[2];

        auto psi = Contract(this->mps.data()[p], this->mps.data()[p + 1]);  // contract

        auto lbl = psi.labels();  // memorize label
        auto psi_T = psi.get_block_();
        psi_T.flatten_();  // flatten to 1d

        cytnx_uint64 new_dim =
          min(min(dim_l * this->mps.phys_dim(p), dim_r * this->mps.phys_dim(p + 1)),
              this->mps.virt_dim());

        // cout << "bkpt1\n";
        //  calculate local ortho_mps:
        // omps = []
        std::vector<Tensor> omps;
        // anet = cytnx.Network("hL_AA_hR.net");
        auto anet = Network();
        anet.FromString({
          "hL: ;-1,1",
          "psi: ;1,2,3,4",
          "hR: ;-2,4",
          "TOUT: ;-1,2,3,-2",
        });

        for (cytnx_int64 ip = 0; ip < this->ortho_mps.size(); ip++) {
          auto opsi = Contract(this->ortho_mps[ip].data()[p], this->ortho_mps[ip].data()[p + 1]);
          opsi.set_rowrank(0);
          anet.PutUniTensors({"hL", "psi", "hR"}, {this->hLRs[ip][p], opsi, this->hLRs[ip][p + 2]});
          auto out = anet.Launch(true).get_block_();
          omps.push_back(out);
          omps.back().flatten_();
        }

        // cout << psi_T << endl;
        auto out = optimize_psi(
          psi_T, {this->LR[p], this->mpo.get_op(p), this->mpo.get_op(p + 1), this->LR[p + 2]},
          maxit, krydim, omps, this->weight);
        psi_T = out[1];
        Entemp = out[0].item();
        // cout << psi_T << endl;

        psi_T.reshape_(dim_l, this->mps.phys_dim(p), this->mps.phys_dim(p + 1),
                       dim_r);  // convert psi back to 4-leg form
        psi = UniTensor(psi_T, false, 2);
        psi.set_labels(lbl);
        // self.Ekeep.append(Entemp);

        auto outU = linalg::Svd_truncate(psi, new_dim);
        auto s = outU[0];
        this->mps.data()[p] = outU[1];
        this->mps.data()[p + 1] = outU[2];

        auto slabel = s.labels();
        s = s / s.get_block_().Norm().item();
        s.set_labels(slabel);

        this->mps.data()[p] = Contract(this->mps.data()[p], s);  // absorb s into next neighbor
        this->mps.S_loc() = p;

        // A[p].print_diagram()
        // A[p+1].print_diagram()

        // update LR from right to left:
        // anet = cytnx.Network("R_AMAH.net")
        anet.FromString(
          {"R: ;-2,-1,-3", "B: 1;-4,-1", "M: ;0,-2,-4,-5", "B_Conj: 2;-5,-3", "TOUT: ;0,1,2"});

        anet.PutUniTensors({"R", "B", "M", "B_Conj"},
                           {this->LR[p + 2], this->mps.data()[p + 1], this->mpo.get_op(p + 1),
                            this->mps.data()[p + 1].Conj()});
        this->LR[p + 1] = anet.Launch(true);

        // update hLR from right to left for excited states:
        // anet = cytnx.Network("hR_AMAH.net")
        anet.FromString({"hR: ;-1,-2", "Bv: 1;-4,-1", "Bp: 2,-4;-2", "TOUT: ;1,2"});

        for (cytnx_int64 ip = 0; ip < this->ortho_mps.size(); ip++) {
          auto omps = this->ortho_mps[ip];
          anet.PutUniTensors({"hR", "Bv", "Bp"}, {this->hLRs[ip][p + 2], this->mps.data()[p + 1],
                                                  omps.data()[p + 1].Conj()});
          this->hLRs[ip][p + 1] = anet.Launch(true);
        }
        // print('Sweep[r->l]: %d/%d, Loc:%d,Energy: %f'%(k,numsweeps,p,Ekeep[-1]))
        if (verbose) std::cout << "Energy: " << std::setprecision(13) << Entemp << std::endl;

      }  // r->l

      this->mps.data()[0].set_rowrank(1);
      auto tout = linalg::Svd(this->mps.data()[0], false, true);
      this->mps.data()[0] = tout[1];
      this->mps.S_loc() = -1;

      // a.2 Optimize from left-to-right:
      /*
          psi:                   Projector:

          --A[p]--A[p+1]--s--              --         --
             |       |                     |    | |    |
                                          LR[p]-M-M-LR[p+1]
                                           |    | |    |
                                           --         --
        b.2 Transfer matrix from left to right :
         LR[0]:       LR[1]:

             ---          ---A[0]---
             |            |    |
             L0-         LR[0]-M----    ......
             |            |    |
             ---          ---A*[0]--

        c.2 For Left to Right, we want A's to be in shape
                   -------------
                  /             \
         virt ____| chi     2   |____ phys
                  |             |
                  |        chi  |____ virt
                  \             /
                   -------------
      */
      for (cytnx_int64 p = 0; p < this->mps.size() - 1; p++) {
        if (verbose) std::cout << "  [>>] upd loc:" << p << "| ";
        cytnx_int64 dim_l = this->mps.data()[p].shape()[0];
        cytnx_int64 dim_r = this->mps.data()[p + 1].shape()[2];

        auto psi = Contract(this->mps.data()[p], this->mps.data()[p + 1]);  // contract
        auto lbl = psi.labels();  // memorize label
        auto psi_T = psi.get_block_();
        psi_T.flatten_();  // flatten to 1d

        cytnx_int64 new_dim =
          min(min(dim_l * this->mps.phys_dim(p), dim_r * this->mps.phys_dim(p + 1)),
              this->mps.virt_dim());

        // calculate local ortho_mps:
        std::vector<Tensor> omps;
        // anet = cytnx.Network("hL_AA_hR.net");
        auto anet = Network();
        anet.FromString({
          "hL: ;-1,1",
          "psi: ;1,2,3,4",
          "hR: ;-2,4",
          "TOUT: ;-1,2,3,-2",
        });

        for (cytnx_int64 ip = 0; ip < this->ortho_mps.size(); ip++) {
          auto opsi = Contract(this->ortho_mps[ip].data()[p], this->ortho_mps[ip].data()[p + 1]);
          opsi.set_rowrank(0);
          anet.PutUniTensors({"hL", "psi", "hR"}, {this->hLRs[ip][p], opsi, this->hLRs[ip][p + 2]});
          omps.push_back(anet.Launch(true).get_block_());
          omps.back().flatten_();
        }

        auto out = optimize_psi(
          psi_T, {this->LR[p], this->mpo.get_op(p), this->mpo.get_op(p + 1), this->LR[p + 2]},
          maxit, krydim, omps, this->weight);
        psi_T = out[1];
        Entemp = out[0].item();
        psi_T.reshape_(dim_l, this->mps.phys_dim(p), this->mps.phys_dim(p + 1),
                       dim_r);  // convert psi back to 4-leg form
        psi = UniTensor(psi_T, false, 2);
        psi.set_labels(lbl);
        // self.Ekeep.append(Entemp);

        auto outU = linalg::Svd_truncate(psi, new_dim);
        auto s = outU[0];
        this->mps.data()[p] = outU[1];
        this->mps.data()[p + 1] = outU[2];
        // s,self.mps.A[p],self.mps.A[p+1] = cytnx.linalg.Svd_truncate(psi,new_dim)

        auto slabel = s.labels();
        s = s / s.get_block_().Norm().item();
        s.set_labels(slabel);

        this->mps.data()[p + 1] =
          Contract(s, this->mps.data()[p + 1]);  // absorb s into next neighbor.
        this->mps.S_loc() = p + 1;

        // anet = cytnx.Network("L_AMAH.net");
        anet.FromString(
          {"L: ;-2,-1,-3", "A: -1,-4;1", "M: ;-2,0,-4,-5", "A_Conj: -3,-5;2", "TOUT: ;0,1,2"});

        anet.PutUniTensors(
          {"L", "A", "A_Conj", "M"},
          {this->LR[p], this->mps.data()[p], this->mps.data()[p].Conj(), this->mpo.get_op(p)});
        this->LR[p + 1] = anet.Launch(true);

        // update hLR when calculate excited state:
        // anet = cytnx.Network("hL_AMAH.net");
        anet.FromString({"hL: ;-1,-2", "Av: -1,-4;1", "Ap: -2,-4;2", "TOUT: ;1,2"});
        for (cytnx_int64 ip = 0; ip < this->ortho_mps.size(); ip++) {
          auto omps = this->ortho_mps[ip];
          anet.PutUniTensors({"hL", "Av", "Ap"},
                             {this->hLRs[ip][p], this->mps.data()[p], omps.data()[p].Conj()});
          this->hLRs[ip][p + 1] = anet.Launch(true);
        }
        // print('Sweep[l->r]: %d of %d, Loc: %d,Energy: %f' % (k, numsweeps, p, Ekeep[-1]))
        if (verbose) std::cout << "Energy: " << std::setprecision(13) << Entemp << std::endl;
      }

      this->mps.data().back().set_rowrank(2);
      tout = linalg::Svd(this->mps.data().back(), true, false);  // last one.
      this->mps.data().back() = tout[1];
      this->mps.S_loc() = this->mps.data().size();

      return Entemp;

    }  // DMRG_impl::sweep

    Scalar DMRG_impl::sweepv2(const bool &verbose, const cytnx_int64 &maxit,
                              const cytnx_int64 &krydim) {
      Scalar Entemp;

      // a. Optimize from right-to-left:
      /*
        psi:                   Projector:

          --A[p]--A[p+1]--s--              --         --
             |       |                     |    | |    |
                                          LR[p]-M-M-LR[p+1]
                                           |    | |    |
                                           --         --
        b. Transfer matrix from right to left :
         LR[-1]:       LR[-2]:

             ---          ---A[-1]---
               |               |    |
             --MR         -----M--LR[-1]   ......
               |               |    |
             ---          ---A*[-1]--

        c. For Right to Left, we want A's to be in shape
                   -------------
                  /             \
         virt ____| chi     chi |____ virt
                  |             |
         phys ____| 2           |
                  \             /
                   -------------
      */

      for (cytnx_int64 p = this->mps.size() - 2; p > -1;
           p--) {  // in range(self.mps.Nsites()-2,-1,-1):
        if (verbose) std::cout << "  [<<] upd loc:" << p << "| " << std::flush;

        cytnx_int64 dim_l = this->mps.data()[p].shape()[0];
        cytnx_int64 dim_r = this->mps.data()[p + 1].shape()[2];

        auto psi = Contract(this->mps.data()[p], this->mps.data()[p + 1]);  // contract
        cytnx_uint64 new_dim =
          min(min(dim_l * this->mps.phys_dim(p), dim_r * this->mps.phys_dim(p + 1)),
              this->mps.virt_dim());

        // cout << "bkpt1\n";
        //  calculate local ortho_mps:
        // omps = []
        std::vector<UniTensor> omps;
        // anet = cytnx.Network("hL_AA_hR.net");
        auto anet = Network();
        anet.FromString({
          "hL: ;-1,1",
          "psi: ;1,2,3,4",
          "hR: ;-2,4",
          "TOUT: -1,2;3,-2",
        });

        for (cytnx_int64 ip = 0; ip < this->ortho_mps.size(); ip++) {
          auto opsi = Contract(this->ortho_mps[ip].data()[p], this->ortho_mps[ip].data()[p + 1]);
          opsi.set_rowrank(0);
          anet.PutUniTensors({"hL", "psi", "hR"}, {this->hLRs[ip][p], opsi, this->hLRs[ip][p + 2]});
          auto out = anet.Launch(true);
          omps.push_back(out);
          // omps.back().flatten_();
        }

        psi.set_rowrank(2);
        // psi.print_diagram();
        // cout << psi << endl;
        auto out = optimize_psi_new(
          psi, {this->LR[p], this->mpo.get_op(p), this->mpo.get_op(p + 1), this->LR[p + 2]}, maxit,
          krydim, omps, this->weight);
        psi = out[1];
        // cout << psi << endl;
        Entemp = out[0].item();
        // psi.print_diagram();
        // exit(1);
        // psi_T.reshape_(dim_l, this->mps.phys_dim(p), this->mps.phys_dim(p+1), dim_r); //convert
        // psi back to 4-leg form psi = UniTensor(psi_T,2); psi.set_labels(lbl);

        // self.Ekeep.append(Entemp);

        auto outU = linalg::Svd_truncate(psi, new_dim);
        auto s = outU[0];
        this->mps.data()[p] = outU[1];
        this->mps.data()[p + 1] = outU[2];

        auto slabel = s.labels();
        s = s / s.get_block_().Norm().item();
        s.set_labels(slabel);

        this->mps.data()[p] = Contract(this->mps.data()[p], s);  // absorb s into next neighbor
        this->mps.S_loc() = p;

        // A[p].print_diagram()
        // A[p+1].print_diagram()

        // update LR from right to left:
        // anet = cytnx.Network("R_AMAH.net")
        anet.FromString(
          {"R: ;-2,-1,-3", "B: 1;-4,-1", "M: ;0,-2,-4,-5", "B_Conj: 2;-5,-3", "TOUT: ;0,1,2"});

        anet.PutUniTensors({"R", "B", "M", "B_Conj"},
                           {this->LR[p + 2], this->mps.data()[p + 1], this->mpo.get_op(p + 1),
                            this->mps.data()[p + 1].Conj()});
        this->LR[p + 1] = anet.Launch(true);

        // update hLR from right to left for excited states:
        // anet = cytnx.Network("hR_AMAH.net")
        anet.FromString({"hR: ;-1,-2", "Bv: 1;-4,-1", "Bp: 2,-4;-2", "TOUT: ;1,2"});

        for (cytnx_int64 ip = 0; ip < this->ortho_mps.size(); ip++) {
          auto omps = this->ortho_mps[ip];
          anet.PutUniTensors({"hR", "Bv", "Bp"}, {this->hLRs[ip][p + 2], this->mps.data()[p + 1],
                                                  omps.data()[p + 1].Conj()});
          this->hLRs[ip][p + 1] = anet.Launch(true);
        }
        // print('Sweep[r->l]: %d/%d, Loc:%d,Energy: %f'%(k,numsweeps,p,Ekeep[-1]))
        if (verbose) std::cout << "Energy: " << std::setprecision(13) << Entemp << std::endl;

      }  // r->l

      this->mps.data()[0].set_rowrank(1);
      auto tout = linalg::Svd(this->mps.data()[0], false, true);
      this->mps.data()[0] = tout[1];
      this->mps.S_loc() = -1;

      // a.2 Optimize from left-to-right:
      /*
          psi:                   Projector:

          --A[p]--A[p+1]--s--              --         --
             |       |                     |    | |    |
                                          LR[p]-M-M-LR[p+1]
                                           |    | |    |
                                           --         --
        b.2 Transfer matrix from left to right :
         LR[0]:       LR[1]:

             ---          ---A[0]---
             |            |    |
             L0-         LR[0]-M----    ......
             |            |    |
             ---          ---A*[0]--

        c.2 For Left to Right, we want A's to be in shape
                   -------------
                  /             \
         virt ____| chi     2   |____ phys
                  |             |
                  |        chi  |____ virt
                  \             /
                   -------------
      */
      for (cytnx_int64 p = 0; p < this->mps.size() - 1; p++) {
        if (verbose) std::cout << "  [>>] upd loc:" << p << "| ";
        cytnx_int64 dim_l = this->mps.data()[p].shape()[0];
        cytnx_int64 dim_r = this->mps.data()[p + 1].shape()[2];

        auto psi = Contract(this->mps.data()[p], this->mps.data()[p + 1]);  // contract
        // auto lbl = psi.labels(); // memorize label
        // auto psi_T = psi.get_block_(); psi_T.flatten_();// flatten to 1d

        cytnx_int64 new_dim =
          min(min(dim_l * this->mps.phys_dim(p), dim_r * this->mps.phys_dim(p + 1)),
              this->mps.virt_dim());

        // calculate local ortho_mps:
        std::vector<UniTensor> omps;
        // anet = cytnx.Network("hL_AA_hR.net");
        auto anet = Network();
        anet.FromString({
          "hL: ;-1,1",
          "psi: ;1,2,3,4",
          "hR: ;-2,4",
          "TOUT: -1,2;3,-2",
        });

        for (cytnx_int64 ip = 0; ip < this->ortho_mps.size(); ip++) {
          auto opsi = Contract(this->ortho_mps[ip].data()[p], this->ortho_mps[ip].data()[p + 1]);
          opsi.set_rowrank(0);
          anet.PutUniTensors({"hL", "psi", "hR"}, {this->hLRs[ip][p], opsi, this->hLRs[ip][p + 2]});
          omps.push_back(anet.Launch(true));
          // omps.back().flatten_();
        }

        psi.set_rowrank(2);
        auto out = optimize_psi_new(
          psi, {this->LR[p], this->mpo.get_op(p), this->mpo.get_op(p + 1), this->LR[p + 2]}, maxit,
          krydim, omps, this->weight);
        psi = out[1];
        Entemp = out[0].item();
        // psi_T.reshape_(dim_l,this->mps.phys_dim(p),this->mps.phys_dim(p+1),dim_r);// convert psi
        // back to 4-leg form psi = UniTensor(psi_T,2); psi.set_labels(lbl);
        // self.Ekeep.append(Entemp);

        auto outU = linalg::Svd_truncate(psi, new_dim);
        auto s = outU[0];
        this->mps.data()[p] = outU[1];
        this->mps.data()[p + 1] = outU[2];
        // s,self.mps.A[p],self.mps.A[p+1] = cytnx.linalg.Svd_truncate(psi,new_dim)

        auto slabel = s.labels();
        s = s / s.get_block_().Norm().item();
        s.set_labels(slabel);

        this->mps.data()[p + 1] =
          Contract(s, this->mps.data()[p + 1]);  // absorb s into next neighbor.
        this->mps.S_loc() = p + 1;

        // anet = cytnx.Network("L_AMAH.net");
        anet.FromString(
          {"L: ;-2,-1,-3", "A: -1,-4;1", "M: ;-2,0,-4,-5", "A_Conj: -3,-5;2", "TOUT: ;0,1,2"});

        anet.PutUniTensors(
          {"L", "A", "A_Conj", "M"},
          {this->LR[p], this->mps.data()[p], this->mps.data()[p].Conj(), this->mpo.get_op(p)});
        this->LR[p + 1] = anet.Launch(true);

        // update hLR when calculate excited state:
        // anet = cytnx.Network("hL_AMAH.net");
        anet.FromString({"hL: ;-1,-2", "Av: -1,-4;1", "Ap: -2,-4;2", "TOUT: ;1,2"});
        for (cytnx_int64 ip = 0; ip < this->ortho_mps.size(); ip++) {
          auto omps = this->ortho_mps[ip];
          anet.PutUniTensors({"hL", "Av", "Ap"},
                             {this->hLRs[ip][p], this->mps.data()[p], omps.data()[p].Conj()});
          this->hLRs[ip][p + 1] = anet.Launch(true);
        }
        // print('Sweep[l->r]: %d of %d, Loc: %d,Energy: %f' % (k, numsweeps, p, Ekeep[-1]))
        if (verbose) std::cout << "Energy: " << std::setprecision(13) << Entemp << std::endl;
      }

      this->mps.data().back().set_rowrank(2);
      tout = linalg::Svd(this->mps.data().back(), true, false);  // last one.
      this->mps.data().back() = tout[1];
      this->mps.S_loc() = this->mps.data().size();

      return Entemp;

    }  // DMRG_impl::sweep

  }  // namespace tn_algo

}  // namespace cytnx
