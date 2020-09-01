#ifndef _xlinalg_cy_H_
#define _xlinalg_cy_H_

#include "TypeConvert.hpp"
#include "cytnx_error.hpp"
//#include "Tensor.hpp"
//#include "Storage.hpp"
#include "CyTensor.hpp"
//#include "LinOp.hpp"
#include <functional>


namespace torcyx{
        namespace xlinalg{
            /*            
            // Add:
            //==================================================
            torcyx::CyTensor Add(const torcyx::CyTensor &Lt, const torcyx::CyTensor &Rt);
            template<class T>
            torcyx::CyTensor Add(const T &lc,const torcyx::CyTensor &Rt);
            template<class T>
            torcyx::CyTensor Add(const torcyx::CyTensor &Lt,const T &rc);
            
            // Sub:
            //==================================================
            torcyx::CyTensor Sub(const torcyx::CyTensor &Lt, const torcyx::CyTensor &Rt);
            template<class T>
            torcyx::CyTensor Sub(const T &lc, const torcyx::CyTensor &Rt);    
            template<class T>
            torcyx::CyTensor Sub(const torcyx::CyTensor &Lt, const T &rc);

            // Mul:
            //==================================================
            torcyx::CyTensor Mul(const torcyx::CyTensor &Lt, const torcyx::CyTensor &Rt);
            template<class T>
            torcyx::CyTensor Mul(const T &lc,const torcyx::CyTensor &Rt);
            template<class T>
            torcyx::CyTensor Mul(const torcyx::CyTensor &Lt,const T &rc);

            // Div:
            //==================================================
            torcyx::CyTensor Div(const torcyx::CyTensor &Lt, const torcyx::CyTensor &Rt);
            template<class T>
            torcyx::CyTensor Div(const T &lc,const torcyx::CyTensor &Rt);
            template<class T>
            torcyx::CyTensor Div(const torcyx::CyTensor &Lt,const T &rc);


            // Mod:
            //==================================================
            torcyx::CyTensor Mod(const torcyx::CyTensor &Lt, const torcyx::CyTensor &Rt);
            template<class T>
            torcyx::CyTensor Mod(const T &lc,const torcyx::CyTensor &Rt);
            template<class T>
            torcyx::CyTensor Mod(const torcyx::CyTensor &Lt,const T &rc);


            std::vector<torcyx::CyTensor> Svd(const torcyx::CyTensor &Tin, const bool &is_U=true, const bool &is_vT=true);
            std::vector<torcyx::CyTensor> Svd_truncate(const torcyx::CyTensor &Tin, const cytnx_uint64 &keepdim, const bool &is_U=true, const bool &is_vT=true);
            std::vector<torcyx::CyTensor> Hosvd(const torcyx::CyTensor &Tin, const std::vector<cytnx_uint64> &mode, const bool &is_core=true, const bool &is_Ls=false,const std::vector<cytnx_int64> &trucate_dim=std::vector<cytnx_int64>());

            torcyx::CyTensor ExpH(const torcyx::CyTensor &Tin, const double &a=1, const double &b=0);
            torcyx::CyTensor ExpM(const torcyx::CyTensor &Tin, const double &a=1, const double &b=0);
            torcyx::CyTensor Trace(const torcyx::CyTensor &Tin, const cytnx_int64 &a, const cytnx_int64 &b, const bool &by_label=false);
            std::vector<torcyx::CyTensor> Qr(const torcyx::CyTensor &Tin, const bool &is_tau=false);
            std::vector<torcyx::CyTensor> Qdr(const torcyx::CyTensor &Tin, const bool &is_tau=false);

            // Pow:
            //==================================================
            CyTensor Pow(const CyTensor &Tin, const double &p);
            
            void Pow_(CyTensor &Tin, const double &p);
            */
        }//xlinalg
        /*
        torcyx::CyTensor operator+(const torcyx::CyTensor &Lt, const torcyx::CyTensor &Rt);
        template<class T>
        torcyx::CyTensor operator+(const T &lc, const torcyx::CyTensor &Rt);
        template<class T>
        torcyx::CyTensor operator+(const torcyx::CyTensor &Lt, const T &rc);
        
        torcyx::CyTensor operator-(const torcyx::CyTensor &Lt, const torcyx::CyTensor &Rt);
        template<class T>
        torcyx::CyTensor operator-(const T &lc, const torcyx::CyTensor &Rt);
        template<class T>
        torcyx::CyTensor operator-(const torcyx::CyTensor &Lt, const T &rc);
        
        torcyx::CyTensor operator*(const torcyx::CyTensor &Lt, const torcyx::CyTensor &Rt);
        template<class T>
        torcyx::CyTensor operator*(const T &lc, const torcyx::CyTensor &Rt);
        template<class T>
        torcyx::CyTensor operator*(const torcyx::CyTensor &Lt, const T &rc);
        
        torcyx::CyTensor operator/(const torcyx::CyTensor &Lt, const torcyx::CyTensor &Rt);
        template<class T>
        torcyx::CyTensor operator/(const T &lc, const torcyx::CyTensor &Rt);
        template<class T>
        torcyx::CyTensor operator/(const torcyx::CyTensor &Lt, const T &rc);

        torcyx::CyTensor operator%(const torcyx::CyTensor &Lt, const torcyx::CyTensor &Rt);
        template<class T>
        torcyx::CyTensor operator%(const T &lc, const torcyx::CyTensor &Rt);
        template<class T>
        torcyx::CyTensor operator%(const torcyx::CyTensor &Lt, const T &rc);
        */

}// torcyx



#endif
