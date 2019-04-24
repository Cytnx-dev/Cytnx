#include "Type.hpp"
#include "tor10_error.hpp"
using namespace std;

std::string tor10::Type::getname(const unsigned int &type_id){


    switch (type_id){
        case this->Void:
            return string("Void");
        case this->ComplexDouble:
            return string("Complex Double (Complex Float64)");
        case this->ComplexFloat:
            return string("Complex Float (Complex Float32)");
        case this->Double:
            return string("Double (Float64)");
        case this->Float:
            return string("Float32");
        case this->Int64:
            return string("Int64");
        case this->Uint64:
            return string("Uint64");
        case this->Int32:
            return string("Int32");
        case this->Uint32:
            return string("Uint32");

        default:
            tor10_error_msg(1,"%s","[ERROR] invalid type");
    }

}
        
namespace tor10{
    Type tor10type;
}
