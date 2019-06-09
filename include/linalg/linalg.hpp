#ifndef _linalg_H_
#define _linalg_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include "Storage.hpp"
#include "linalg/linalg_internal_interface.hpp"


namespace cytnx{
    /**
    @namespace cytnx::linalg
    @brief linear algebra related functions.
    */
    namespace linalg{
        // Add:
        Tensor Add(const Tensor &Lt, const Tensor &Rt);
        template<class T>
        Tensor Add(const T &lc, const Tensor &Rt);    
        template<class T>
        Tensor Add(const Tensor &Lt, const T &rc);

        // Sub:
        Tensor Sub(const Tensor &Lt, const Tensor &Rt);
        template<class T>
        Tensor Sub(const T &lc, const Tensor &Rt);    
        template<class T>
        Tensor Sub(const Tensor &Lt, const T &rc);

        // Mul:
        Tensor Mul(const Tensor &Lt, const Tensor &Rt);
        template<class T>
        Tensor Mul(const T &lc, const Tensor &Rt);    
        template<class T>
        Tensor Mul(const Tensor &Lt, const T &rc);

        // Div:
        Tensor Div(const Tensor &Lt, const Tensor &Rt);
        template<class T>
        Tensor Div(const T &lc, const Tensor &Rt);    
        template<class T>
        Tensor Div(const Tensor &Lt, const T &rc);


        // Svd:
        /** 
        @brief Perform Singular-Value decomposition on a rank-2 Tensor.
        @param Tin a \link cytnx::Tensor Tensor \endlink, it should be a rank-2 tensor (matrix)
        @param is_U if return a left uniform matrix.
        @param is_vT if return a right uniform matrix.
        @return [std::vector<Tensors>]  

            1. the first tensor is a 1-d tensor contanin the singular values
            2. the second tensor is the left uniform matrix [U], a 2-d tensor (matrix). It only return when is_U=true.
            3. the third tensor is the right uniform matrix [vT], a 2-d tensor (matrix). It only return when is_vT=true.
        */
        std::vector<Tensor> Svd(const Tensor &Tin, const bool &is_U=true, const bool &is_vT=true);

        // Eig:
        std::vector<Tensor> Eigh(const Tensor &Tin, const bool &is_V=false);

        // Matmul:
        Tensor Matmul(const Tensor &Tin, const Tensor &Tout);


        // Inv:
        Tensor Inv(const Tensor &Tin);
        void Inv_(Tensor &Tin);


        // Conj:
        Tensor Conj(const Tensor &Tin);
        void Conj_(Tensor &Tin);

        // Exp:
        Tensor Exp(const Tensor &Tin);
        void Exp_(Tensor &Tin);

        // Diag:
        Tensor Diag(const Tensor &Tin);
        


    }// namespace linalg
    

    // operators:
    Tensor operator+(const Tensor &Lt, const Tensor &Rt);
    template<class T>
    Tensor operator+(const T &lc, const Tensor &Rt);
    template<class T>
    Tensor operator+(const Tensor &Lt, const T &rc);
    //------------------------------------
    Tensor operator-(const Tensor &Lt, const Tensor &Rt);
    template<class T>
    Tensor operator-(const T &lc, const Tensor &Rt);
    template<class T>
    Tensor operator-(const Tensor &Lt, const T &rc);
    //-----------------------------------
    Tensor operator*(const Tensor &Lt, const Tensor &Rt);
    template<class T>
    Tensor operator*(const T &lc, const Tensor &Rt);
    template<class T>
    Tensor operator*(const Tensor &Lt, const T &rc);
    //----------------------------------
    Tensor operator/(const Tensor &Lt, const Tensor &Rt);
    template<class T>
    Tensor operator/(const T &lc, const Tensor &Rt);
    template<class T>
    Tensor operator/(const Tensor &Lt, const T &rc);



}


#endif
