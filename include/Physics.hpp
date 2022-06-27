#ifndef _Physics_H_
#define _Physics_H_
#include "Type.hpp"
#include "Device.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
#include <vector>
#include <initializer_list>
#include <string>
namespace cytnx {
  namespace physics {
    //@{
    /**
    @brief create Spin-S representation matrix.
    @param S the Spin index. should be multiple of 1/2.
    @param Comp the Component of Spin, it can be 'x', 'y' and 'z'
    @param device the device that the Tensor is put on. It can be any device defined in \link
    cytnx::Device cytnx::Device \endlink

    @return
        [Tensor]

    */
    Tensor spin(const cytnx_double &S, const std::string &Comp, const int &device = Device.cpu);

    /// @cond
    Tensor spin(const cytnx_double &S, const char &Comp, const int &device = Device.cpu);
    /// @endcond

    //@{
    /**
    @brief create Pauli matrix.
    @param Comp the Component of Spin, it can be 'x', 'y' and 'z'
    @param device the device that the Tensor is put on. It can be any device defined in \link
    cytnx::Device cytnx::Device \endlink

    @return
        [Tensor]

    */
    Tensor pauli(const std::string &Comp, const int &device = Device.cpu);

    /// @cond
    Tensor pauli(const char &Comp, const int &device = Device.cpu);
    /// @endcond

#define kBoltz cytnx_double(1.380649e-23)  // J/K
#define hPlanck cytnx_double(6.62607015e-34)  // J-s
#define hBar cytnx_double(1.05457181e-34)  // J-s

  }  // namespace physics
}  // namespace cytnx

/// @cond
namespace cytnx {
  namespace qgates {
    UniTensor pauli_x(const int &device = Device.cpu);
    UniTensor pauli_y(const int &device = Device.cpu);
    UniTensor pauli_z(const int &device = Device.cpu);
    UniTensor hadamard(const int &device = Device.cpu);
    UniTensor phase_shift(const cytnx_double &phase, const int &device = Device.cpu);
    UniTensor swap(const int &device = Device.cpu);
    UniTensor sqrt_swap(const int &device = Device.cpu);
    UniTensor toffoli(const int &device = Device.cpu);

    UniTensor cntl_gate_2q(const UniTensor &gate_1q);

  }  // namespace qgates
}  // namespace cytnx
/// @endcond

#endif
