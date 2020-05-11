#include "cytnx.hpp"
#include <iostream>
using namespace cytnx;
using namespace std;
namespace cyx = cytnx_extension;

int main(int argc, char* argv[]){
    cyx::Network N;
    N.Fromfile("example.net");
    cout << N << endl;   
}
