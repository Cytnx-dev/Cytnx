#include "linalg_internal_interface.hpp"

using namespace std;
linalg_internal_interface lii;

linalg_internal_interface::linalg_internal_interface(){
    arithmic_internal = vector<vector<Arithmicfunc_oii> >(N_Type,vector<Arithmicfunc_oii>(N_Type));
}

