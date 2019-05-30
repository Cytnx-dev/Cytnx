#ifndef _H_Symmetry
#define _H_Symmetry
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "intrusive_ptr_base.hpp"
#include <string>
#include <cstdio>
namespace cytnx{

    class SymmetryType_class{
        public:
            enum : int{
                U=-1,
                Z=0,
            };
            std::string getname(const int &stype);
    };
    extern SymmetryType_class SymType;


    class Symmetry_base: public intrusive_ptr_base<Symmetry_base>{
        public:
            int stype_id;
            int n; 
            Symmetry_base(){};
            Symmetry_base(const int &n){
                this->Init(n);
            };
            Symmetry_base(const Symmetry_base &rhs);
            Symmetry_base& operator=(const Symmetry_base &rhs);
      
            virtual void Init(const int &n){};
            virtual boost::intrusive_ptr<Symmetry_base> copy(){};
            virtual bool check_qnum(const cytnx_int64 &in_qnum); // check the passed in qnums satisfy the symmetry requirement.
            //virtual std::vector<cytnx_int64>& combine_rule(const std::vector<cytnx_int64> &inL, const std::vector<cytnx_int64> &inR);
    };

    class U1Symmetry : public Symmetry_base{
        public:
            U1Symmetry(){};
            U1Symmetry(const int &n){this->Init(n);};
            void Init(const int &n){
                this->stype_id = SymType.U;      
                this->n = n;
                if(n!=0) cytnx_error_msg(1,"%s","[ERROR] U1Symmetry should set n = 0");
            }        
            boost::intrusive_ptr<Symmetry_base> copy(){
                boost::intrusive_ptr<Symmetry_base> out(new U1Symmetry(this->n));
                return out;
            }
            bool check_qnum(const cytnx_int64 &in_qnum);
    
    };
    class ZnSymmetry : public Symmetry_base{
       public:
            ZnSymmetry(){};
            ZnSymmetry(const int &n){this->Init(n);};
            void Init(const int &n){
                this->stype_id = SymType.Z;
                this->n = n;
                if(n<=1) cytnx_error_msg(1,"%s","[ERROR] ZnSymmetry can only have n > 1");
            }
            boost::intrusive_ptr<Symmetry_base> copy(){
                boost::intrusive_ptr<Symmetry_base> out(new ZnSymmetry(this->n));
                return out;
            }
            bool check_qnum(const cytnx_int64 &in_qnum);
    };


    //=====================================
    // this is API
    class Symmetry{
        public:
            boost::intrusive_ptr<Symmetry_base> _impl;
            
            //Symmetry() : _impl(new U1Symmetry()){}; //default is U1Symmetry
            Symmetry(const int &stype=-1, const int &n=0){
                this->Init(stype,n);
            }; //default is U1Symmetry

            // genenrators:
            static Symmetry U1(){
                return Symmetry(SymType.U);
            }
            static Symmetry Zn(const int &n){
                return Symmetry(SymType.Z,n);
            }

            Symmetry copy(){
                Symmetry out;
                out._impl = this->_impl->copy();
                return out;
            }

            Symmetry& operator=(const Symmetry &rhs){
               this->_impl = rhs._impl->copy(); // let's enforce copy now
            }
            Symmetry(const  Symmetry &rhs){
                this->_impl = rhs._impl->copy(); // let's enforce copy now
            }

            int & stype_id() const {
                return this->_impl->stype_id;
            }

            int & n() const{
                return this->_impl->n;
            }

            const std::string stype(){
                return SymType.getname(this->_impl->stype_id) + std::to_string(this->_impl->n);
            }

            void astype(const int &stype, const int &n){
                this->Init(stype,n);
            }
            void Init(const int &stype=-1, const int &n=0){
                if(stype==SymType.U){
                    boost::intrusive_ptr<Symmetry_base> tmp(new U1Symmetry(n));
                    this->_impl = tmp;
                }else if(stype==SymType.Z){
                    boost::intrusive_ptr<Symmetry_base> tmp(new ZnSymmetry(n));
                    this->_impl = tmp;
                }else{
                    cytnx_error_msg(1,"%s","[ERROR] invalid symmetry type.");
                }
            }

            // this serves as generator!!
            bool check_qnum(const cytnx_int64 &qnum){
                return this->_impl->check_qnum(qnum);
            }

    };

    bool operator==(Symmetry &lhs, Symmetry &rhs);

}

#endif
