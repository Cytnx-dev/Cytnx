#include "Storage.hpp"


#include <iostream>

using namespace std;

namespace cytnx{

    std::ostream& operator<<(std::ostream& os, Storage &in){
        in.print(); 
        return os; 
    }



}





