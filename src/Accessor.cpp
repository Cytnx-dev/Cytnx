#include "Accessor.hpp"
#include "utils/str_utils.hpp"
#include<iostream>
using namespace std;
namespace cytnx{

    Accessor::Accessor(const cytnx_int64 &loc){
        this->_type = Accessor::Singl;
        this->loc  = loc; 
    } 

    // all constr. ( use string to dispatch )            
    Accessor::Accessor(const std::string &str){
        //std::cout << str << "|" << std::endl;
        if((str=="all") || (str==":"))
            this->_type = Accessor::All;
        else{
            cytnx_error_msg(true,"[ERROR] only Accessor::all() can use string to init.%s","\n");
            //std::vector<std::string> token = str_split(str,false,":");
            
            //cytnx_error_msg(token.size()<=1,"[ERROR] no ':' in resolving accessor. use integer directly.%s","\n");
            //cytnx_error_msg(token.size()>3,"[ERROR] invalid string to Accessor, make sure no space exist.%s","\n");
            //this->min = token[0].size()==0?0,std::stoll(token[0]);
            //this->max = token[1].size()==0?std::stoll(token[1]);
            //if(token.size()==3)
            //    this->step = std::stoll(token[2]);
            //else
            //    this->step = 1;
            //this->_type = Accessor::Range;

            //cout << token << endl;
        }
    }

    // range constr. 
    Accessor::Accessor(const cytnx_int64 &min, const cytnx_int64 &max, const cytnx_int64 &step){
        cytnx_error_msg(step==0,"[ERROR] cannot have step=0 for range%s","\n");
        this->_type = Accessor::Range;
        this->min = min;
        this->max = max;
        this->step = step;
    }


    //copy constructor:
    Accessor::Accessor(const Accessor& rhs){
        this->_type = rhs._type;
        this->min  = rhs.min;
        this->max  = rhs.max;
        this->loc  = rhs.loc;
        this->step = rhs.step;
    }

    //copy assignment:
    Accessor& Accessor::operator=(const Accessor& rhs){
        this->_type = rhs._type;
        this->min  = rhs.min;
        this->max  = rhs.max;
        this->loc  = rhs.loc;
        this->step = rhs.step;
        return *this;
    }

    // get the real len from dim
    // if _type is all, pos will be null, and len == dim
    // if _type is range, pos will be the locator, and len == len(pos)
    // if _type is singl, pos will be pos, and len == 0 
    void Accessor::get_len_pos(const cytnx_uint64 &dim, cytnx_uint64 &len, std::vector<cytnx_uint64> &pos) const{
        #ifdef UNI_DEBUG 
        cytnx_error_msg(this->_type==Accessor::none,"%s","[DEBUG][ERROR] try to call get_len from an un-initialize Accessor.");
        #endif

        pos.clear();

        if(this->_type == Accessor::All){
            len = dim;
        }else if(this->_type == Accessor::Range){
            cytnx_int64 r_min=this->min,r_max=this->max;
            cytnx_error_msg((r_max-r_min)/this->step<0,"%s","[ERROR] upper bound and larger bound inconsistent with step sign");
            len = (r_max-r_min)/this->step;
            //std::cout << len << " " << dim << std::endl;
            //if((r_max-r_min)%this->step) len+=1;
            
            for(cytnx_int64 i = r_min; i!=r_max; i+=this->step){
                pos.push_back((i%dim+dim)%dim);
                //std::cout << pos.back() << std::endl;
            }
            
        }else if(this->_type == Accessor::Singl){
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
    //============================================
    std::ostream& operator<<(std::ostream& os, const Accessor &in){
        if(in.type()==Accessor::Singl){
            os << in.loc;
        }else if(in.type()==Accessor::All){
            os << ":";
        }else if(in.type()==Accessor::Range){
            os << in.min << ":" << in.max << ":" << in.step;
        }else{
            cytnx_error_msg(true,"[ERROR][cout] Accessor is Void!%s","\n");
        }
        return os;
    }


}//namespace cytnx
