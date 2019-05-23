#ifndef _linalg_H_
#define _linalg_H_

#include "Type.hpp"
#include "tor10_error.hpp"
#include "Tensor.hpp"
#include "Storage.hpp"
#include "linalg/linalg_internal_interface.hpp"


namespace tor10{

    // Add:
    Tensor Add(const Tensor &Lt, const Tensor &Rt);
    Tensor operator+(const Tensor &Lt, const Tensor &Rt);

    template<class T>
    Tensor Add(const T &lc, const Tensor &Rt);    
    template<class T>
    Tensor operator+(const T &lc, const Tensor &Rt){
        return Add(lc,Rt);
    }

    template<class T>
    Tensor Add(const Tensor &Lt, const T &rc);
    template<class T>
    Tensor operator+(const Tensor &Lt, const T &rc){
       return Add(Lt,rc);
    }

    // Sub:
    Tensor Sub(const Tensor &Lt, const Tensor &Rt);
    Tensor operator-(const Tensor &Lt, const Tensor &Rt);

    template<class T>
    Tensor Sub(const T &lc, const Tensor &Rt);    
    template<class T>
    Tensor operator-(const T &lc, const Tensor &Rt){
        return Sub(lc,Rt);
    }

    template<class T>
    Tensor Sub(const Tensor &Lt, const T &rc);
    template<class T>
    Tensor operator-(const Tensor &Lt, const T &rc){
       return Sub(Lt,rc);
    }

    // Mul:
    Tensor Mul(const Tensor &Lt, const Tensor &Rt);
    Tensor operator*(const Tensor &Lt, const Tensor &Rt);

    template<class T>
    Tensor Mul(const T &lc, const Tensor &Rt);    
    template<class T>
    Tensor operator*(const T &lc, const Tensor &Rt){
        return Mul(lc,Rt);
    }

    template<class T>
    Tensor Mul(const Tensor &Lt, const T &rc);
    template<class T>
    Tensor operator*(const Tensor &Lt, const T &rc){
       return Mul(Lt,rc);
    }

    // Div:
    Tensor Div(const Tensor &Lt, const Tensor &Rt);
    Tensor operator/(const Tensor &Lt, const Tensor &Rt);

    template<class T>
    Tensor Div(const T &lc, const Tensor &Rt);    
    template<class T>
    Tensor operator/(const T &lc, const Tensor &Rt){
        return Div(lc,Rt);
    }

    template<class T>
    Tensor Div(const Tensor &Lt, const T &rc);
    template<class T>
    Tensor operator/(const Tensor &Lt, const T &rc){
       return Div(Lt,rc);
    }
}


#endif
