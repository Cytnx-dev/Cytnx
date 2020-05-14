#ifndef _Physics_H_
#define _Physics_H_
#include "Type.hpp"
#include "Device.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include <vector>
#include <initializer_list>
#include <string>
namespace cytnx{
    namespace physics{
        //@{
        /**
        @brief create Spin-S representation matrix. 
        @param S the Spin index. should be multiple of 1/2.
        @param Comp the Component of Spin, it can be 'x', 'y' and 'z'
        @param device the device that the Tensor is put on. It can be any device defined in \link cytnx::Device cytnx::Device \endlink

        @return 
            [Tensor]
       
        */    
        Tensor spin(const cytnx_double &S, const std::string &Comp, const int &device=Device.cpu);

        /// @cond 
        Tensor spin(const cytnx_double &S, const char &Comp, const int &device=Device.cpu);
        /// @endcond

        //@{
        /**
        @brief create Pauli matrix. 
        @param Comp the Component of Spin, it can be 'x', 'y' and 'z'
        @param device the device that the Tensor is put on. It can be any device defined in \link cytnx::Device cytnx::Device \endlink

        @return 
            [Tensor]
       
        */    
        Tensor pauli(const std::string &Comp, const int &device=Device.cpu);

        /// @cond 
        Tensor pauli(const char &Comp, const int &device=Device.cpu);
        /// @endcond
    }
}

#endif
