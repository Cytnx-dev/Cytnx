#ifndef _H_Storage_
#define _H_Storage_

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <typeinfo>
#include <vector>
#include <complex>

#ifdef UNI_OMP
    #include <omp.h>
#endif

#include "Type.hpp"
#include "Device.hpp"
#include "intrusive_ptr_base.hpp"
#include "tor10_error.hpp"



namespace tor10{

    class Storage_base : public intrusive_ptr_base<Storage_base> {
        public:
            void* Mem;
            //std::vector<unsigned int> shape;

            unsigned long long len; // default 0
            unsigned int dtype_id; // default 0, Void
            int device; // default -1, on cpu

            Storage_base(): len(0), Mem(NULL),dtype_id(0), device(-1){};
            //Storage_base(const std::initializer_list<unsigned int> &init_shape);
            //Storage_base(const std::vector<unsigned int> &init_shape);    
            Storage_base(const unsigned long long &len_in,const int &device);

            Storage_base(Storage_base &Rhs);
            Storage_base& operator=(Storage_base &Rhs);
            boost::intrusive_ptr<Storage_base> astype(const unsigned int &dtype);
            

            //void Init(const std::initializer_list<unsigned int> &init_shape);
            std::string dtype();

            ~Storage_base();


            template<class T>
            T& at(const unsigned int &idx);
            
            template<class T>   
            T* data();


            /*
                This function is design to check the type mismatch. 
                Handy for developer to exclude the assign of double 
                C pointer into a non-DoubleStorage.
                
                For example:
                float *cptr = (float*)calloc(4,sizeof(float));

                intrusive_ptr<Storage> array(new DoubleStorage());
                array->_Init_byptr((void*)cptr,4); // This is fatal, since we alloc cptr as float, 
                                                   // but apon free, DoubleStorage will free 2x 
                                                   // of memory!!!!

                array->_Init_byptr_safe(cptr,4);   // This is design to avoid the above problem 
                                                   // by checking the type of input pointer with 
                                                   // the type of Storage before call _Init_byptr.
                                                   // [Note] this will intorduce overhead!!.

            */
            template<class T>
            void _Init_byptr_safe(T *rawptr, const unsigned long long &len_in){
                //check:
                if(this->dtype_id==tor10type.Float){
                        tor10_error_msg(typeid(T) != typeid(float),"%s","[ERROR _Init_byptr_safe type not match]");
                }else if(this->dtype_id==tor10type.Double){
                        tor10_error_msg(typeid(T) != typeid(double),"%s","[ERROR _Init_byptr_safe type not match]");
                }else{
                    tor10_error_msg(1,"[FATAL] ERROR%s","\n");
                }

                this->_Init_byptr((void*)rawptr,len_in);
            }


            // these is the one that do the work, and customize with Storage_base
            //virtual void Init(const std::vector<unsigned int> &init_shape);
            virtual void Init(const unsigned long long &len_in, const int &device=-1);
            virtual void _Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device=-1);

            // this function will return a new storage with the same type as the one 
            // that initiate this function. 
            virtual boost::intrusive_ptr<Storage_base> _create_new_sametype();

            // [future] this will move the memory to device / cpu
            virtual void to_(const int &device);
            virtual boost::intrusive_ptr<Storage_base> to(const int &device);

            virtual boost::intrusive_ptr<Storage_base> copy();

            // this will perform permute on the underlying memory. 
            virtual boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper);
            virtual void Move_memory_(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper); 
            virtual void PrintElem_byShape(std::ostream& os, const std::vector<tor10_uint64> &shape, const std::vector<tor10_uint64> &mapper={});        


    };        

                    
    class FloatStorage : public Storage_base{
        public:
            FloatStorage(){this->dtype_id=tor10type.Float;};
            void Init(const unsigned long long &len_in, const int &device=-1);
            void _Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device=-1);
            boost::intrusive_ptr<Storage_base> _create_new_sametype();
            boost::intrusive_ptr<Storage_base> copy();
            boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper);
            void Move_memory_(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper); 
            void to_(const int &device);
            boost::intrusive_ptr<Storage_base> to(const int &device);
            void PrintElem_byShape(std::ostream& os, const std::vector<tor10_uint64> &shape, const std::vector<tor10_uint64> &mapper={});        
    };          

    class DoubleStorage: public Storage_base{
        public:
            DoubleStorage(){this->dtype_id=tor10type.Double;};
            void Init(const unsigned long long &len_in,const int &device=-1);
            void _Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device=-1);
            boost::intrusive_ptr<Storage_base> _create_new_sametype();
            boost::intrusive_ptr<Storage_base> copy();
            boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper);
            void Move_memory_(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper); 
            void to_(const int &device);
            boost::intrusive_ptr<Storage_base> to(const int &device);
            void PrintElem_byShape(std::ostream& os, const std::vector<tor10_uint64> &shape, const std::vector<tor10_uint64> &mapper={});        
    };



    class ComplexDoubleStorage: public Storage_base{
        public:
            ComplexDoubleStorage(){this->dtype_id=tor10type.ComplexDouble;};
            void Init(const unsigned long long &len_in, const int &device=-1);
            void _Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device=-1);
            boost::intrusive_ptr<Storage_base> _create_new_sametype();
            boost::intrusive_ptr<Storage_base> copy();
            boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper);
            void Move_memory_(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper); 
            void to_(const int &device);
            boost::intrusive_ptr<Storage_base> to(const int &device);
            void PrintElem_byShape(std::ostream& os, const std::vector<tor10_uint64> &shape, const std::vector<tor10_uint64> &mapper={});        
    };

    class ComplexFloatStorage: public Storage_base{
        public:
            ComplexFloatStorage(){this->dtype_id=tor10type.ComplexFloat;};
            void Init(const unsigned long long &len_in,const int &device=-1);
            void _Init_byptr(void *rawptr, const unsigned long long &len_in,const int &device=-1);
            boost::intrusive_ptr<Storage_base> _create_new_sametype();
            boost::intrusive_ptr<Storage_base> copy();
            boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper);
            void Move_memory_(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper); 
            void to_(const int &device);
            boost::intrusive_ptr<Storage_base> to(const int &device);
            void PrintElem_byShape(std::ostream& os, const std::vector<tor10_uint64> &shape, const std::vector<tor10_uint64> &mapper={});        
    };

    class Int64Storage : public Storage_base{
        public:
            Int64Storage(){this->dtype_id=tor10type.Int64;};
            void Init(const unsigned long long &len_in, const int &device=-1);
            void _Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device=-1);
            boost::intrusive_ptr<Storage_base> _create_new_sametype();
            boost::intrusive_ptr<Storage_base> copy();
            boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper);
            void Move_memory_(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper); 
            void to_(const int &device);
            boost::intrusive_ptr<Storage_base> to(const int &device);
            void PrintElem_byShape(std::ostream& os, const std::vector<tor10_uint64> &shape, const std::vector<tor10_uint64> &mapper={});        
    };          

    class Uint64Storage : public Storage_base{
        public:
            Uint64Storage(){this->dtype_id=tor10type.Uint64;};
            void Init(const unsigned long long &len_in, const int &device=-1);
            void _Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device=-1);
            boost::intrusive_ptr<Storage_base> _create_new_sametype();
            boost::intrusive_ptr<Storage_base> copy();
            boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper);
            void Move_memory_(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper); 
            void to_(const int &device);
            boost::intrusive_ptr<Storage_base> to(const int &device);
            void PrintElem_byShape(std::ostream& os, const std::vector<tor10_uint64> &shape, const std::vector<tor10_uint64> &mapper={});        
    };          

    class Int32Storage : public Storage_base{
        public:
            Int32Storage(){this->dtype_id=tor10type.Int32;};
            void Init(const unsigned long long &len_in, const int &device=-1);
            void _Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device=-1);
            boost::intrusive_ptr<Storage_base> _create_new_sametype();
            boost::intrusive_ptr<Storage_base> copy();
            boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper);
            void Move_memory_(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper); 
            void to_(const int &device);
            boost::intrusive_ptr<Storage_base> to(const int &device);
            void PrintElem_byShape(std::ostream& os, const std::vector<tor10_uint64> &shape, const std::vector<tor10_uint64> &mapper={});        
    };          

    class Uint32Storage : public Storage_base{
        public:
            Uint32Storage(){this->dtype_id=tor10type.Uint32;};
            void Init(const unsigned long long &len_in, const int &device=-1);
            void _Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device=-1);
            boost::intrusive_ptr<Storage_base> _create_new_sametype();
            boost::intrusive_ptr<Storage_base> copy();
            boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper);
            void Move_memory_(const std::vector<tor10_uint64> &old_shape, const std::vector<tor10_uint64> &mapper, const std::vector<tor10_uint64> &invmapper); 
            void to_(const int &device);
            boost::intrusive_ptr<Storage_base> to(const int &device);
            void PrintElem_byShape(std::ostream& os, const std::vector<tor10_uint64> &shape, const std::vector<tor10_uint64> &mapper={});        
    };          

    typedef boost::intrusive_ptr<Storage_base> (*pStorage_init)();
    class Storage_init_interface: public Type{
        public:
            std::vector<pStorage_init> USIInit;
            Storage_init_interface();
    };
    extern Storage_init_interface Storage_init;
}

#endif
