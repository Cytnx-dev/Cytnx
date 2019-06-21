#include "cytnx.hpp"
#include <iostream>


using namespace cytnx;
using namespace std;
int main(){ 

    Tensor A = arange(60);

    Tensor B = A.reshape({5,12});
    cout << A << endl;
    cout << B << endl;

    return 0;
}

