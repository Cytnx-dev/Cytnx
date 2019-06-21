#include "cytnx.hpp"
#include <iostream>


using namespace cytnx;
using namespace std;
int main(){ 

    Tensor A({3,4,5});
    cout << A.shape() << endl;

    Tensor B = A.permute({0,2,1});
    cout << B.shape() << endl;


    return 0;
}

