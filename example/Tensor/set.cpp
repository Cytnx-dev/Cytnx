#include "cytnx.hpp"
#include <iostream>


using namespace cytnx;
using namespace std;
int main(){

    typedef Accessor ac;

    Tensor A = arange(60).reshape({3,4,5});
    cout << A << endl;

    // 1. Set with Tensor
    Tensor B = zeros({4,3});
    cout << B << endl;
    A.set({ac(2),ac::all(),ac::range(2,5,1)},B);
    cout << A << endl;

    // 2. Set with constant.
    A.set({ac(0),ac::all(),ac::range(0,2,1)},999);
    cout << A << endl;


    return 0;
}
