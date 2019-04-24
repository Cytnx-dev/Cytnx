#include "Symmetry.hpp"
#include <string>
using namespace std;


std::string tor10::SymmetryType::getname(const int &stype_id){
    if(stype_id==this->U){
        return string("U");
    }else if(stype_id==this->Z){
        return string("Z");
    }else{
        tor10_error_msg(true,"%s","[ERROR] invalid stype_id");
    }
    // extend more in here!!
}


bool tor10::operator==(Symmetry &lhs, Symmetry &rhs){
    return(lhs.stype_id() == rhs.stype_id()) && (lhs.n() == rhs.n());
}




bool tor10::Symmetry_base::check_qnum(const tor10_int64 &qnum){
    tor10_error_msg(1,"%s","[ERROR][Internal] should not all Symmerty base!");
}

///=========================
bool tor10::U1Symmetry::check_qnum(const tor10_int64 &qnum){
    return true;
}

///========================
bool tor10::ZnSymmetry::check_qnum(const tor10_int64 &qnum){
    return ( qnum >= 0 ) && (qnum < this->n);
}

//++++++++++++++++++++++++
namespace tor10{
    SymmetryType tor10stype;
};
