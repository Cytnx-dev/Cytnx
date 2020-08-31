#ifndef _xlinalg_cy_H_
#define _xlinalg_cy_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
//#include "Tensor.hpp"
//#include "Storage.hpp"
#include "CyTensor.hpp"
//#include "LinOp.hpp"
#include <functional>


namespace cytnx{
    namespace ml{
        namespace xlinalg{
            /*            
            // Add:
            //==================================================
            cytnx::ml::CyTensor Add(const cytnx::ml::CyTensor &Lt, const cytnx::ml::CyTensor &Rt);
            template<class T>
            cytnx::ml::CyTensor Add(const T &lc,const cytnx::ml::CyTensor &Rt);
            template<class T>
            cytnx::ml::CyTensor Add(const cytnx::ml::CyTensor &Lt,const T &rc);
            
            // Sub:
            //==================================================
            cytnx::ml::CyTensor Sub(const cytnx::ml::CyTensor &Lt, const cytnx::ml::CyTensor &Rt);
            template<class T>
            cytnx::ml::CyTensor Sub(const T &lc, const cytnx::ml::CyTensor &Rt);    
            template<class T>
            cytnx::ml::CyTensor Sub(const cytnx::ml::CyTensor &Lt, const T &rc);

            // Mul:
            //==================================================
            cytnx::ml::CyTensor Mul(const cytnx::ml::CyTensor &Lt, const cytnx::ml::CyTensor &Rt);
            template<class T>
            cytnx::ml::CyTensor Mul(const T &lc,const cytnx::ml::CyTensor &Rt);
            template<class T>
            cytnx::ml::CyTensor Mul(const cytnx::ml::CyTensor &Lt,const T &rc);

            // Div:
            //==================================================
            cytnx::ml::CyTensor Div(const cytnx::ml::CyTensor &Lt, const cytnx::ml::CyTensor &Rt);
            template<class T>
            cytnx::ml::CyTensor Div(const T &lc,const cytnx::ml::CyTensor &Rt);
            template<class T>
            cytnx::ml::CyTensor Div(const cytnx::ml::CyTensor &Lt,const T &rc);


            // Mod:
            //==================================================
            cytnx::ml::CyTensor Mod(const cytnx::ml::CyTensor &Lt, const cytnx::ml::CyTensor &Rt);
            template<class T>
            cytnx::ml::CyTensor Mod(const T &lc,const cytnx::ml::CyTensor &Rt);
            template<class T>
            cytnx::ml::CyTensor Mod(const cytnx::ml::CyTensor &Lt,const T &rc);


            std::vector<cytnx::ml::CyTensor> Svd(const cytnx::ml::CyTensor &Tin, const bool &is_U=true, const bool &is_vT=true);
            std::vector<cytnx::ml::CyTensor> Svd_truncate(const cytnx::ml::CyTensor &Tin, const cytnx_uint64 &keepdim, const bool &is_U=true, const bool &is_vT=true);
            std::vector<cytnx::ml::CyTensor> Hosvd(const cytnx::ml::CyTensor &Tin, const std::vector<cytnx_uint64> &mode, const bool &is_core=true, const bool &is_Ls=false,const std::vector<cytnx_int64> &trucate_dim=std::vector<cytnx_int64>());

            cytnx::ml::CyTensor ExpH(const cytnx::ml::CyTensor &Tin, const double &a=1, const double &b=0);
            cytnx::ml::CyTensor ExpM(const cytnx::ml::CyTensor &Tin, const double &a=1, const double &b=0);
            cytnx::ml::CyTensor Trace(const cytnx::ml::CyTensor &Tin, const cytnx_int64 &a, const cytnx_int64 &b, const bool &by_label=false);
            std::vector<cytnx::ml::CyTensor> Qr(const cytnx::ml::CyTensor &Tin, const bool &is_tau=false);
            std::vector<cytnx::ml::CyTensor> Qdr(const cytnx::ml::CyTensor &Tin, const bool &is_tau=false);

            // Pow:
            //==================================================
            CyTensor Pow(const CyTensor &Tin, const double &p);
            
            void Pow_(CyTensor &Tin, const double &p);
            */
        }//xlinalg
        /*
        cytnx::ml::CyTensor operator+(const cytnx::ml::CyTensor &Lt, const cytnx::ml::CyTensor &Rt);
        template<class T>
        cytnx::ml::CyTensor operator+(const T &lc, const cytnx::ml::CyTensor &Rt);
        template<class T>
        cytnx::ml::CyTensor operator+(const cytnx::ml::CyTensor &Lt, const T &rc);
        
        cytnx::ml::CyTensor operator-(const cytnx::ml::CyTensor &Lt, const cytnx::ml::CyTensor &Rt);
        template<class T>
        cytnx::ml::CyTensor operator-(const T &lc, const cytnx::ml::CyTensor &Rt);
        template<class T>
        cytnx::ml::CyTensor operator-(const cytnx::ml::CyTensor &Lt, const T &rc);
        
        cytnx::ml::CyTensor operator*(const cytnx::ml::CyTensor &Lt, const cytnx::ml::CyTensor &Rt);
        template<class T>
        cytnx::ml::CyTensor operator*(const T &lc, const cytnx::ml::CyTensor &Rt);
        template<class T>
        cytnx::ml::CyTensor operator*(const cytnx::ml::CyTensor &Lt, const T &rc);
        
        cytnx::ml::CyTensor operator/(const cytnx::ml::CyTensor &Lt, const cytnx::ml::CyTensor &Rt);
        template<class T>
        cytnx::ml::CyTensor operator/(const T &lc, const cytnx::ml::CyTensor &Rt);
        template<class T>
        cytnx::ml::CyTensor operator/(const cytnx::ml::CyTensor &Lt, const T &rc);

        cytnx::ml::CyTensor operator%(const cytnx::ml::CyTensor &Lt, const cytnx::ml::CyTensor &Rt);
        template<class T>
        cytnx::ml::CyTensor operator%(const T &lc, const cytnx::ml::CyTensor &Rt);
        template<class T>
        cytnx::ml::CyTensor operator%(const cytnx::ml::CyTensor &Lt, const T &rc);
        */
    }// ml

}// cytnx



#endif
