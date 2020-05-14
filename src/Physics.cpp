#include "Physics.hpp"
#include "Storage.hpp"
#include "Generator.hpp"
#include "utils/utils.hpp"
#include "utils/utils_internal_interface.hpp"
#include "linalg.hpp"
#include <cfloat>
#include <iostream>
#include <cmath>
using namespace std;
namespace cytnx{
    namespace physics{
        Tensor spin(const cytnx_double &S, const std::string &Comp, const int &device){
            cytnx_error_msg(S<0.5,"[ERROR][physics::spin] S can only be multiple of 1/2.%s","\n");        
            // dim
            cytnx_double tN = S*2;
            cytnx_double Intp,Fracp; Fracp = modf(tN,&Intp);
            cytnx_error_msg(Fracp > 1.0e-10,"[ERROR][physics::spin] S can only be multiple of 1/2.%s","\n");
            cytnx_uint64 Dim = tN + 1;

            Tensor Out = zeros({Dim,Dim},Type.ComplexDouble,device);

            // direction:
            if(Comp == "z" || Comp == "Z"){
                for(cytnx_uint64 a=0; a<Dim; a++){
                    Out.at<cytnx_complex128>({a,a}) = S - a;
                }                        
            }else if(Comp == "y" || Comp == "Y"){
                for(cytnx_uint64 a=0; a<Dim; a++){
                    if(a!=0)
                        Out.at<cytnx_complex128>({a,a-1}) = cytnx_complex128(0,1)*pow((S + 1) * (2 * a) - (a + 1) * a,0.5) / 2;
                    if(a!= Dim-1)
                        Out.at<cytnx_complex128>({a,a+1}) = cytnx_complex128(0,-1)*pow((S + 1) * (2 * a + 2) - (a + 2) * (a + 1),0.5) / 2;
                }                        
            }else if(Comp == "x" || Comp == "X"){
                for(cytnx_uint64 a=0; a<Dim; a++){
                    if(a!=0)
                        Out.at<cytnx_complex128>({a,a-1}) = pow((S + 1)*(2 * a) - (a + 1)*a,0.5)/2;
                    if(a!= Dim-1)
                        Out.at<cytnx_complex128>({a,a+1}) = pow((S + 1) * (2 * a + 2) - (a + 2) * (a + 1),0.5) / 2;
                }                        
            }else{
                cytnx_error_msg(true,"[ERROR][physics::spin] Invalid Component, can only be 'x', 'y' or 'z'.%s","\n");
            }     

            return Out;


        }
        Tensor spin(const cytnx_double &S, const char &Comp, const int &device){
            return spin(S,string(1,Comp),device);
        }

        Tensor pauli(const std::string &Comp, const int &device){
            Tensor Out = zeros(4,Type.ComplexDouble,device);
            
            if(Comp == "z" || Comp == "Z"){
                Out.at<cytnx_complex128>({0,0}) = 1;
                Out.at<cytnx_complex128>({1,1}) = -1;
            }else if(Comp == "x" || Comp == "X"){
                Out.at<cytnx_complex128>({0,1}) = 1;
                Out.at<cytnx_complex128>({1,0}) = 1;
            }else if(Comp == "y" || Comp == "Y"){
                Out.at<cytnx_complex128>({0,1}) = cytnx_complex128(0,-1);
                Out.at<cytnx_complex128>({1,0}) = cytnx_complex128(0, 1);
            }else{
                cytnx_error_msg(true,"[ERROR][physics::pauli] Invalid Component, can only be 'x', 'y' or 'z'.%s","\n");
            }
            return Out;
        }
        Tensor pauli(const char &Comp, const int &device){
            return pauli(string(1,Comp),device);
        }


    }
}
