#include "Accessor.hpp"

#include<iostream>
namespace cytnx{

    Accessor::Accessor(const cytnx_int64 &loc){
        this->type = Accessor::Singl;
        this->loc  = loc; 
    } 

    // all constr. ( use string to dispatch )            
    Accessor::Accessor(const std::string &str){
        this->type = Accessor::All;
    }

    // range constr. 
    Accessor::Accessor(const cytnx_int64 &min, const cytnx_int64 &max, const cytnx_int64 &step){
        cytnx_error_msg(step==0,"[ERROR] cannot have step=0 for range%s","\n");
        this->type = Accessor::Range;
        this->min = min;
        this->max = max;
        this->step = step;
    }


    //copy constructor:
    Accessor::Accessor(const Accessor& rhs){
        this->type = rhs.type;
        this->min  = rhs.min;
        this->max  = rhs.max;
        this->loc  = rhs.loc;
        this->step = rhs.step;
    }

    //copy assignment:
    Accessor& Accessor::operator=(const Accessor& rhs){
        this->type = rhs.type;
        this->min  = rhs.min;
        this->max  = rhs.max;
        this->loc  = rhs.loc;
        this->step = rhs.step;
        return *this;
    }

    // get the real len from dim
    // if type is all, pos will be null, and len == dim
    // if type is range, pos will be the locator, and len == len(pos)
    // if type is singl, pos will be pos, and len == 0 
    void Accessor::get_len_pos(const cytnx_uint64 &dim, cytnx_uint64 &len, std::vector<cytnx_uint64> &pos) const{
        #ifdef UNI_DEBUG 
        cytnx_error_msg(this->type==Accessor::none,"%s","[DEBUG][ERROR] try to call get_len from an un-initialize Accessor.");
        #endif

        pos.clear();

        if(this->type == Accessor::All){
            len = dim;
        }else if(this->type == Accessor::Range){
            cytnx_int64 r_min=this->min,r_max=this->max;
            cytnx_error_msg((r_max-r_min)/this->step<0,"%s","[ERROR] upper bound and larger bound inconsistent with step sign");
            len = (r_max-r_min)/this->step;
            //std::cout << len << " " << dim << std::endl;
            //if((r_max-r_min)%this->step) len+=1;
            
            for(cytnx_int64 i = r_min; i!=r_max; i+=this->step){
                pos.push_back((i%dim+dim)%dim);
                //std::cout << pos.back() << std::endl;
            }
            
        }else if(this->type == Accessor::Singl){
            //check:
            //std::cout << this->loc << " " << dim << std::endl;
            cytnx_error_msg(std::abs(this->loc) >= dim,"[ERROR] index is out of bound%s","\n");
            len = 1;
            if(this->loc<0)
                pos.push_back(this->loc+dim);
            else
                pos.push_back(this->loc);
        }
    }

}//namespace cytnx
