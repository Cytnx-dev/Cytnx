#ifndef _random_H_
#define _random_H_
#include "Type.hpp"
#include "Device.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include "Storage.hpp"
#include <vector>
#include <initializer_list>
#include <random>


namespace cytnx{
    namespace random{

        // Make_normal:
        //==================================================
        /** 
        @brief Randomize the memory of a Tensor with normal distributon
        @param Tin a \link cytnx::Tensor Tensor \endlink
        @param mean the mean of a normal distribution
        @param std the standard deviation (sigma) of a normal distribution. 
        @param seed the seed for the random generator. [Default] Using device entropy.

            [Note] The Tensor sould be real floating type or complex type.  

        */
        void Make_normal(Storage &Sin, const double &mean, const double &std, const unsigned int &seed=std::random_device()());

        // Make_normal:
        //==================================================
        /** 
        @brief Randomize the memory of a Storage with normal distributon
        @param Tin a \link cytnx::Storage Storage \endlink
        @param mean the mean of a normal distribution
        @param std the standard deviation (sigma) of a normal distribution. 
        @param seed the seed for the random generator. [Default] Using device entropy.

            [Note] The Storage sould be real floating type or complex type.  

        */
        void Make_normal(Tensor  &Tin, const double &mean, const double &std, const unsigned int &seed=std::random_device()());


        // normal:
        //@{
        /**
        @brief create an Tensor with all the elements are initialized with normal distribution
        @param Nelem the number of elements     
        @param mean the mean of the normal distribution
        @param std the standard deviation (sigma) of the normal distribution
        @param device the device that the Tensor is put on. It can be any device defined in \link cytnx::Device cytnx::Device \endlink
        @param seed the seed for the random generator. [Default] using device entropy

        @return 
            [Tensor]
       
        */    
        Tensor normal(const cytnx_uint64 &Nelem, const double &mean, const double &std, const int &device=Device.cpu, const unsigned int &seed=std::random_device()());

        /**
        @brief create an Tensor with all the elements are initialized with normal distribution
        @param Nelem the shape of the Tensor               
        @param mean the mean of the normal distribution
        @param std the standard deviation (sigma) of the normal distribution
        @param device the device that the Tensor is put on. It can be any device defined in \link cytnx::Device cytnx::Device \endlink
        @param seed the seed for the random generator. [Default] using device entropy

        @return 
            [Tensor]
       
        */    
        Tensor normal(const std::vector<cytnx_uint64> &Nelem, const double &mean, const double &std, const int &device=Device.cpu, const unsigned int &seed=std::random_device()());
        //@}    



    }
}

#endif
