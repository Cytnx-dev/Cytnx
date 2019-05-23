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
    Tensor operator+(const Tensor &Lc, const Tensor &Rc);

    template<class T>
    Tensor Add(const T &lc, const Tensor &Rt);    
    template<class T>
    Tensor operator+(const T &lc, const Tensor &Rc){
        return Add(lc,Rc);
    }

    template<class T>
    Tensor Add(const Tensor &Lc, const T &rc);
    template<class T>
    Tensor operator+(const Tensor &Lc, const T &rc){
       return Add(Lc,rc);
    }

    // Sub:
    Tensor Sub(const Tensor &Lt, const Tensor &Rt);
    Tensor operator-(const Tensor &Lc, const Tensor &Rc);

    template<class T>
    Tensor Sub(const T &lc, const Tensor &Rt);    
    template<class T>
    Tensor operator-(const T &lc, const Tensor &Rc){
        return Sub(lc,Rc);
    }

    template<class T>
    Tensor Sub(const Tensor &Lc, const T &rc);
    template<class T>
    Tensor operator-(const Tensor &Lc, const T &rc){
       return Sub(Lc,rc);
    }

    // Mul:
    Tensor Mul(const Tensor &Lt, const Tensor &Rt);
    Tensor operator*(const Tensor &Lc, const Tensor &Rc);

    template<class T>
    Tensor Mul(const T &lc, const Tensor &Rt);    
    template<class T>
    Tensor operator*(const T &lc, const Tensor &Rc){
        return Mul(lc,Rc);
    }

    template<class T>
    Tensor Mul(const Tensor &Lc, const T &rc);
    template<class T>
    Tensor operator*(const Tensor &Lc, const T &rc){
       return Mul(Lc,rc);
    }

    // Div:
    Tensor Div(const Tensor &Lt, const Tensor &Rt);
    Tensor operator/(const Tensor &Lc, const Tensor &Rc);

    template<class T>
    Tensor Div(const T &lc, const Tensor &Rt);    
    template<class T>
    Tensor operator/(const T &lc, const Tensor &Rc){
        return Div(lc,Rc);
    }

    template<class T>
    Tensor Div(const Tensor &Lc, const T &rc);
    template<class T>
    Tensor operator/(const Tensor &Lc, const T &rc){
       return Div(Lc,rc);
    }
}


#endif
