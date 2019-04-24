#include "linalg_internal_interface.hpp"

using namespace std;
linalg_internal_interface lii;

linalg_internal_interface::linalg_internal_interface(){
    add_internal = vector<vector<Addfunc_oii> >(N_Type,vector<Addfunc_oii>(N_Type));
}

