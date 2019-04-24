#ifndef _H_Bond_
#define _H_Bond_
#include "Type.hpp"
#include "Symmetry.hpp"
#include "tor10_error.hpp"
#include <initializer_list>
#include <vector>
namespace tor10{

    enum bondType: int{
        BD_BRA = -1,
        BD_KET = 1,
        BD_REGULAR =0
    };

    class Bond{
        private:
            tor10_uint64 _dim;
            bondType _type;
            std::vector<std::vector<tor10_int64> > _qnums;
            std::vector<Symmetry> _syms;

        public:

            Bond(): _type(bondType::BD_REGULAR) {};   
            Bond(tor10_uint64 dim, const std::initializer_list<std::initializer_list<tor10_int64> > &in_qnums, const std::initializer_list<Symmetry> &in_syms={});
            Bond(tor10_uint64 dim, const std::vector<std::vector<tor10_int64> > &in_qnums, const std::vector<Symmetry> &in_syms={});

            //Initialize with non-sym
            Bond(tor10_uint64 dim){
                tor10_error_msg(dim==0,"%s","[ERROR] Bond cannot have 0 dimension.");
                this->_type = bondType::BD_REGULAR;
            }

            bondType                                get_type() const& {return this->_type;};
            std::vector<std::vector<tor10_int64> > get_qnums() const& {return this->_qnums;};
            tor10_uint64                                 dim() const &{return this->_dim;};
            tor10_uint32                                Nsym() const &{return this->_syms.size();};
            std::vector<Symmetry>                   get_syms() const &{return this->_syms;};


            void set_type(const bondType &new_bondType){
                this->_type = new_bondType;
            }

            void clear_type(){
                this->_type = bondType::BD_REGULAR;
            }

            Bond(const Bond &rhs){
                this->_dim = rhs.dim();
                this->_type = rhs.get_type();
                this->_qnums = rhs.get_qnums();
                this->_syms  = rhs.get_syms();
            }

            Bond& operator=(const Bond &rhs){
                this->_dim = rhs.dim();
                this->_type = rhs.get_type();
                this->_qnums = rhs.get_qnums();
                this->_syms  = rhs.get_syms();
                return *this;
            }

        

    };//Bond class

    std::ostream& operator<<(std::ostream &os,const Bond &bin);

}



#endif
