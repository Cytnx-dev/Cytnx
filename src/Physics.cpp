#include "Physics.hpp"

#include "Generator.hpp"
#include "utils/utils.hpp"

#include "linalg.hpp"
#include <cfloat>
#include <iostream>
#include <cmath>
using namespace std;

#ifdef BACKEND_TORCH
#else

  #include "backend/Storage.hpp"

namespace cytnx {
  namespace physics {
    Tensor spin(const cytnx_double &S, const std::string &Comp, const int &device) {
      cytnx_error_msg(S < 0.5, "[ERROR][physics::spin] S can only be multiple of 1/2.%s", "\n");
      // dim
      cytnx_double tN = S * 2;
      cytnx_double Intp, Fracp;
      Fracp = modf(tN, &Intp);
      cytnx_error_msg(Fracp > 1.0e-10, "[ERROR][physics::spin] S can only be multiple of 1/2.%s",
                      "\n");
      cytnx_uint64 Dim = tN + 1;

      Tensor Out = zeros({Dim, Dim}, Type.ComplexDouble, device);

      // direction:
      if (Comp == "z" || Comp == "Z") {
        for (cytnx_uint64 a = 0; a < Dim; a++) {
          Out.at<cytnx_complex128>({a, a}) = S - a;
        }
      } else if (Comp == "y" || Comp == "Y") {
        for (cytnx_uint64 a = 0; a < Dim; a++) {
          if (a != 0)
            Out.at<cytnx_complex128>({a, a - 1}) =
              cytnx_complex128(0, 1) * pow((S + 1) * (2 * a) - (a + 1) * a, 0.5) / 2;
          if (a != Dim - 1)
            Out.at<cytnx_complex128>({a, a + 1}) =
              cytnx_complex128(0, -1) * pow((S + 1) * (2 * a + 2) - (a + 2) * (a + 1), 0.5) / 2;
        }
      } else if (Comp == "x" || Comp == "X") {
        for (cytnx_uint64 a = 0; a < Dim; a++) {
          if (a != 0)
            Out.at<cytnx_complex128>({a, a - 1}) = pow((S + 1) * (2 * a) - (a + 1) * a, 0.5) / 2;
          if (a != Dim - 1)
            Out.at<cytnx_complex128>({a, a + 1}) =
              pow((S + 1) * (2 * a + 2) - (a + 2) * (a + 1), 0.5) / 2;
        }
      } else {
        cytnx_error_msg(
          true, "[ERROR][physics::spin] Invalid Component, can only be 'x', 'y' or 'z'.%s", "\n");
      }

      return Out;
    }
    Tensor spin(const cytnx_double &S, const char &Comp, const int &device) {
      return spin(S, string(1, Comp), device);
    }

    Tensor pauli(const std::string &Comp, const int &device) {
      Tensor Out = zeros({2, 2}, Type.ComplexDouble, device);

      if (Comp == "z" || Comp == "Z") {
        Out.at<cytnx_complex128>({0, 0}) = 1;
        Out.at<cytnx_complex128>({1, 1}) = -1;
      } else if (Comp == "x" || Comp == "X") {
        Out.at<cytnx_complex128>({0, 1}) = 1;
        Out.at<cytnx_complex128>({1, 0}) = 1;
      } else if (Comp == "y" || Comp == "Y") {
        Out.at<cytnx_complex128>({0, 1}) = cytnx_complex128(0, -1);
        Out.at<cytnx_complex128>({1, 0}) = cytnx_complex128(0, 1);
      } else {
        cytnx_error_msg(
          true, "[ERROR][physics::pauli] Invalid Component, can only be 'x', 'y' or 'z'.%s", "\n");
      }
      return Out;
    }
    Tensor pauli(const char &Comp, const int &device) { return pauli(string(1, Comp), device); }

  }  // namespace physics
}  // namespace cytnx

namespace cytnx {
  namespace qgates {
    using namespace cytnx;
    UniTensor pauli_x(const int &device) {
      Tensor tmp = cytnx::physics::pauli('x', device);
      return UniTensor(tmp, false, 1);
    }
    UniTensor pauli_y(const int &device) {
      Tensor tmp = cytnx::physics::pauli('y', device);
      return UniTensor(tmp, false, 1);
    }
    UniTensor pauli_z(const int &device) {
      Tensor tmp = cytnx::physics::pauli('z', device);
      return UniTensor(tmp, false, 1);
    }
    UniTensor hadamard(const int &device) {
      Tensor tmp = cytnx::physics::pauli('z', device);
      tmp[{0, 1}] = 1;
      tmp[{1, 0}] = 1;
      return UniTensor(tmp, false, 1);
    }
    UniTensor phase_shift(const cytnx_double &phase, const int &device) {
      Tensor tmp = physics::pauli('z', device);
      tmp[{1, 1}] = exp(cytnx_complex128(0, phase));
      return UniTensor(tmp, false, 1);
    }

    UniTensor swap(const int &device) {
      Tensor tmp = zeros({4, 4}, Type.ComplexDouble, device);
      tmp[{0, 0}] = tmp[{3, 3}] = tmp[{1, 2}] = tmp[{2, 1}] = 1;
      tmp.reshape_({2, 2, 2, 2});
      return UniTensor(tmp, false, 2);
    }

    UniTensor sqrt_swap(const int &device) {
      Tensor tmp = zeros({4, 4}, Type.ComplexDouble, device);
      tmp[{0, 0}] = tmp[{3, 3}] = 1;
      tmp[{1, 1}] = tmp[{2, 2}] = 0.5 * cytnx_complex128(1, 1);
      tmp[{1, 2}] = tmp[{2, 1}] = 0.5 * cytnx_complex128(1, -1);
      tmp.reshape_({2, 2, 2, 2});
      return UniTensor(tmp, false, 2);
    }

    UniTensor toffoli(const int &device) {
      Tensor tmp = zeros({8, 8}, Type.Double, device);
      tmp[{0, 0}] = tmp[{1, 1}] = tmp[{2, 2}] = tmp[{3, 3}] = tmp[{4, 4}] = tmp[{5, 5}] = 1;
      tmp[{6, 7}] = tmp[{7, 6}] = 1;
      tmp.reshape_({2, 2, 2, 2, 2, 2});
      return UniTensor(tmp, false, 3);
    }

    UniTensor cntl_gate_2q(const UniTensor &gate_1q) {
      Tensor tmp = zeros({4, 4}, gate_1q.dtype(), gate_1q.device());
      tmp[{0, 0}] = tmp[{1, 1}] = 1;

      auto gt = gate_1q.get_block_();

      tmp[{2, 2}] = gt[{0, 0}];
      tmp[{2, 3}] = gt[{0, 1}];
      tmp[{3, 2}] = gt[{1, 0}];
      tmp[{3, 3}] = gt[{1, 1}];

      tmp.reshape_({2, 2, 2, 2});
      return UniTensor(tmp, false, 2);
    }

  }  // namespace qgates

  namespace operators {

    UniTensor Sz_shalf(const int &device, const bool &conserve_qns) {
      if (conserve_qns) {
        Bond bd_phy(BD_IN, {Qs(1) >> 1, Qs(-1) >> 1});
        auto out = UniTensor({bd_phy, bd_phy.redirect()});
        out.at({0, 0, 0, 0}) = 0.5;
        out.at({1, 1, 0, 0}) = -0.5;
        return out;

      } else {
        return UniTensor(physics::pauli("z", device).real());
      }
    }
    UniTensor Sp_shalf(const int &device, const bool &conserve_qns, const std::string &aux_dir) {
      if (conserve_qns) {
        Bond bd_phy(BD_IN, {Qs(1) >> 1, Qs(-1) >> 1});
        Bond bd_aux;
        if (aux_dir == "L" or aux_dir == "l")
          bd_aux.Init(BD_IN, {Qs(1) >> 1});
        else if (aux_dir == "R" or aux_dir == "r")
          bd_aux.Init(BD_OUT, {Qs(-1) >> 1});
        else
          cytnx_error_msg(
            true, "[ERROR] aux_dir can only be L or R. (letter capital insensitive)%s", "\n");

        auto out = UniTensor({bd_phy, bd_phy.redirect(), bd_aux});
        out.at({0, 1, 0}) = 1;
        return out;

      } else {
        auto out = UniTensor(zeros({2, 2}, Type.Double, device));
        out.at({0, 1}) = 1;
        return out;
      }
    }
    UniTensor Sn_shalf(const int &device, const bool &conserve_qns, const std::string &aux_dir) {
      if (conserve_qns) {
        Bond bd_phy(BD_IN, {Qs(1) >> 1, Qs(-1) >> 1});
        Bond bd_aux;
        if (aux_dir == "L" or aux_dir == "l")
          bd_aux.Init(BD_IN, {Qs(-1) >> 1});
        else if (aux_dir == "R" or aux_dir == "r")
          bd_aux.Init(BD_OUT, {Qs(1) >> 1});
        else
          cytnx_error_msg(
            true, "[ERROR] aux_dir can only be L or R. (letter capital insensitive)%s", "\n");

        auto out = UniTensor({bd_phy, bd_phy.redirect(), bd_aux});
        out.at({1, 0, 0}) = 1;
        return out;

      } else {
        auto out = UniTensor(zeros({2, 2}, Type.Double, device));
        out.at({1, 0}) = 1;
        return out;
      }
    }

  }  // namespace operators

}  // namespace cytnx
#endif
