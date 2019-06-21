#include "cytnx.hpp"
#include <iostream>


using namespace cytnx;
using namespace std;
int main(){ 

    Tensor A = arange(60);
    cout << A << endl;

    A.reshape_({5,12});
    cout << A << endl;

    return 0;
}

