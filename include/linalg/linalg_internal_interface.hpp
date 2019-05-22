#ifndef _H_linalg_internal_interface_
#define _H_linalg_internal_interface_
#include <iostream>
#include <vector>
#include "Type.hpp"

typedef void (*Arithmicfunc_oii)(void*,void*,void*,const unsigned long long & len, const char &type);



class linalg_internal_interface{

    public:
        std::vector<std::vector<Arithmicfunc_oii>> arithmic_internal;

    linalg_internal_interface();
        


};
extern linalg_internal_interface lii;

#endif

