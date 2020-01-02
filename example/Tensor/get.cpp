#include "cytnx.hpp"
#include <iostream>


using namespace cytnx;
using namespace std;
int main(){

    typedef Accessor ac;

    Tensor A = arange(60).reshape({3,4,5});
    cout << A << endl;

    Tensor B = A.get({ac(2),ac::all(),ac::range(2,5,1)});
    cout << B << endl;


    Tensor B2 = A[{2,ac::all(),ac::range(2,5,1)}];
    cout << B2 << endl;

    return 0;
}
