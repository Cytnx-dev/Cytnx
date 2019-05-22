#include "Storage.hpp"


#include <iostream>

using namespace std;

namespace tor10{

    std::ostream& operator<<(std::ostream& os, Storage &in){
        in.print(); 
        return os; 
    }



}





