#ifndef _linalg_H_
#define _linalg_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include "Storage.hpp"
#include "linalg/linalg_internal_interface.hpp"


namespace cytnx{

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
        std::vector<Tensor> Svd(const Tensor &Tin, const bool &is_U=true, const bool &is_vT=true);

        // Inv:
        Tensor Inv(const Tensor &Tin);
        void Inv_(Tensor &Tin);


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
