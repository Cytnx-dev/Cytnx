#ifndef _H_linalg_internal_interface_
#define _H_linalg_internal_interface_
#include <iostream>
#include <vector>
#include "Type.hpp"

typedef void (*Addfunc_oii)(void*,void*,void*,const unsigned long long & len);



class linalg_internal_interface{

    public:
        std::vector<std::vector<Addfunc_oii>> add_internal;

    linalg_internal_interface();
        


};
extern linalg_internal_interface lii;

#endif

