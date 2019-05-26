#include "Symmetry.hpp"
#include <string>
using namespace std;


std::string cytnx::SymmetryType::getname(const int &stype_id){
    if(stype_id==this->U){
        return string("U");
    }else if(stype_id==this->Z){
        return string("Z");
    }else{
        cytnx_error_msg(true,"%s","[ERROR] invalid stype_id");
    }
    // extend more in here!!
}


bool cytnx::operator==(Symmetry &lhs, Symmetry &rhs){
    return(lhs.stype_id() == rhs.stype_id()) && (lhs.n() == rhs.n());
}




bool cytnx::Symmetry_base::check_qnum(const cytnx_int64 &qnum){
    cytnx_error_msg(1,"%s","[ERROR][Internal] should not all Symmerty base!");
}

///=========================
bool cytnx::U1Symmetry::check_qnum(const cytnx_int64 &qnum){
    return true;
}

///========================
bool cytnx::ZnSymmetry::check_qnum(const cytnx_int64 &qnum){
    return ( qnum >= 0 ) && (qnum < this->n);
}

//++++++++++++++++++++++++
namespace cytnx{
    SymmetryType cytnxstype;
};
