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
#include "cytnx_error.hpp"



namespace cytnx{

    
    class Storage_base : public intrusive_ptr_base<Storage_base> {
        public:
            void* Mem;
            //std::vector<unsigned int> shape;

            unsigned long long len; // default 0
            unsigned int dtype; // default 0, Void
            int device; // default -1, on cpu

            Storage_base(): len(0), Mem(NULL),dtype(0), device(-1){};
            //Storage_base(const std::initializer_list<unsigned int> &init_shape);
            //Storage_base(const std::vector<unsigned int> &init_shape);    
            Storage_base(const unsigned long long &len_in,const int &device);

            Storage_base(Storage_base &Rhs);
            Storage_base& operator=(Storage_base &Rhs);
            boost::intrusive_ptr<Storage_base> astype(const unsigned int &dtype);
            

            //void Init(const std::initializer_list<unsigned int> &init_shape);
            std::string dtype_str() const ;
            std::string device_str() const;
            const unsigned long long &size() const{
                return this->len;
            }
            ~Storage_base();


            template<class T>
            T& at(const unsigned int &idx) const;
            
            template<class T>   
            T* data() const;

            void print();
            void print_info();
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
                if(this->dtype==cytnxtype.Float){
                        cytnx_error_msg(typeid(T) != typeid(float),"%s","[ERROR _Init_byptr_safe type not match]");
                }else if(this->dtype==cytnxtype.Double){
                        cytnx_error_msg(typeid(T) != typeid(double),"%s","[ERROR _Init_byptr_safe type not match]");
                }else{
                    cytnx_error_msg(1,"[FATAL] ERROR%s","\n");
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

            virtual boost::intrusive_ptr<Storage_base> clone();

            // this will perform permute on the underlying memory. 
            virtual boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper);
            virtual void Move_memory_(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper); 
            virtual void PrintElem_byShape(std::ostream& os, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper={});        
            virtual void print_elems();
            virtual void GetElem_byShape(boost::intrusive_ptr<Storage_base> &out, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &len, const std::vector<std::vector<cytnx_uint64> > &locators);

            //generators:
            virtual void set_zeros();
            

    };        

    ///////////////////////////////////                    
    class FloatStorage : public Storage_base{
        public:
            FloatStorage(){this->dtype=cytnxtype.Float;};
            void Init(const unsigned long long &len_in, const int &device=-1);
            void _Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device=-1);
            boost::intrusive_ptr<Storage_base> _create_new_sametype();
            boost::intrusive_ptr<Storage_base> clone();
            boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper);
            void Move_memory_(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper); 
            void to_(const int &device);
            boost::intrusive_ptr<Storage_base> to(const int &device);
            void PrintElem_byShape(std::ostream& os, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper={});        
            void print_elems();
            void GetElem_byShape(boost::intrusive_ptr<Storage_base> &out, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &len, const std::vector<std::vector<cytnx_uint64> > &locators);

            //generators:
            void set_zeros();

    };          

    class DoubleStorage: public Storage_base{
        public:
            DoubleStorage(){this->dtype=cytnxtype.Double;};
            void Init(const unsigned long long &len_in,const int &device=-1);
            void _Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device=-1);
            boost::intrusive_ptr<Storage_base> _create_new_sametype();
            boost::intrusive_ptr<Storage_base> clone();
            boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper);
            void Move_memory_(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper); 
            void to_(const int &device);
            boost::intrusive_ptr<Storage_base> to(const int &device);
            void PrintElem_byShape(std::ostream& os, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper={});        
            void print_elems();
            void GetElem_byShape(boost::intrusive_ptr<Storage_base> &out, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &len, const std::vector<std::vector<cytnx_uint64> > &locators);

            //generators:
            void set_zeros();

    };



    class ComplexDoubleStorage: public Storage_base{
        public:
            ComplexDoubleStorage(){this->dtype=cytnxtype.ComplexDouble;};
            void Init(const unsigned long long &len_in, const int &device=-1);
            void _Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device=-1);
            boost::intrusive_ptr<Storage_base> _create_new_sametype();
            boost::intrusive_ptr<Storage_base> clone();
            boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper);
            void Move_memory_(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper); 
            void to_(const int &device);
            boost::intrusive_ptr<Storage_base> to(const int &device);
            void PrintElem_byShape(std::ostream& os, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper={});        
            void print_elems();
            void GetElem_byShape(boost::intrusive_ptr<Storage_base> &out, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &len, const std::vector<std::vector<cytnx_uint64> > &locators);

            //generators:
            void set_zeros();


    };

    class ComplexFloatStorage: public Storage_base{
        public:
            ComplexFloatStorage(){this->dtype=cytnxtype.ComplexFloat;};
            void Init(const unsigned long long &len_in,const int &device=-1);
            void _Init_byptr(void *rawptr, const unsigned long long &len_in,const int &device=-1);
            boost::intrusive_ptr<Storage_base> _create_new_sametype();
            boost::intrusive_ptr<Storage_base> clone();
            boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper);
            void Move_memory_(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper); 
            void to_(const int &device);
            boost::intrusive_ptr<Storage_base> to(const int &device);
            void PrintElem_byShape(std::ostream& os, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper={});        
            void print_elems();
            void GetElem_byShape(boost::intrusive_ptr<Storage_base> &out, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &len, const std::vector<std::vector<cytnx_uint64> > &locators);

            //generators:
            void set_zeros();


    };

    class Int64Storage : public Storage_base{
        public:
            Int64Storage(){this->dtype=cytnxtype.Int64;};
            void Init(const unsigned long long &len_in, const int &device=-1);
            void _Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device=-1);
            boost::intrusive_ptr<Storage_base> _create_new_sametype();
            boost::intrusive_ptr<Storage_base> clone();
            boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper);
            void Move_memory_(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper); 
            void to_(const int &device);
            boost::intrusive_ptr<Storage_base> to(const int &device);
            void PrintElem_byShape(std::ostream& os, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper={});        
            void print_elems();
            void GetElem_byShape(boost::intrusive_ptr<Storage_base> &out, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &len, const std::vector<std::vector<cytnx_uint64> > &locators);

            //generators:
            void set_zeros();


    };          

    class Uint64Storage : public Storage_base{
        public:
            Uint64Storage(){this->dtype=cytnxtype.Uint64;};
            void Init(const unsigned long long &len_in, const int &device=-1);
            void _Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device=-1);
            boost::intrusive_ptr<Storage_base> _create_new_sametype();
            boost::intrusive_ptr<Storage_base> clone();
            boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper);
            void Move_memory_(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper); 
            void to_(const int &device);
            boost::intrusive_ptr<Storage_base> to(const int &device);
            void PrintElem_byShape(std::ostream& os, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper={});        
            void print_elems();
            void GetElem_byShape(boost::intrusive_ptr<Storage_base> &out, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &len, const std::vector<std::vector<cytnx_uint64> > &locators);


            //generators:
            void set_zeros();



    };          

    class Int32Storage : public Storage_base{
        public:
            Int32Storage(){this->dtype=cytnxtype.Int32;};
            void Init(const unsigned long long &len_in, const int &device=-1);
            void _Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device=-1);
            boost::intrusive_ptr<Storage_base> _create_new_sametype();
            boost::intrusive_ptr<Storage_base> clone();
            boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper);
            void Move_memory_(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper); 
            void to_(const int &device);
            boost::intrusive_ptr<Storage_base> to(const int &device);
            void PrintElem_byShape(std::ostream& os, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper={});        
            void print_elems();
            void GetElem_byShape(boost::intrusive_ptr<Storage_base> &out, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &len, const std::vector<std::vector<cytnx_uint64> > &locators);

            //generators:
            void set_zeros();


    };          


    class Uint32Storage : public Storage_base{
        public:
            Uint32Storage(){this->dtype=cytnxtype.Uint32;};
            void Init(const unsigned long long &len_in, const int &device=-1);
            void _Init_byptr(void *rawptr, const unsigned long long &len_in, const int &device=-1);
            boost::intrusive_ptr<Storage_base> _create_new_sametype();
            boost::intrusive_ptr<Storage_base> clone();
            boost::intrusive_ptr<Storage_base> Move_memory(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper);
            void Move_memory_(const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper); 
            void to_(const int &device);
            boost::intrusive_ptr<Storage_base> to(const int &device);
            void PrintElem_byShape(std::ostream& os, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper={});        
            void print_elems();
            void GetElem_byShape(boost::intrusive_ptr<Storage_base> &out, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &len, const std::vector<std::vector<cytnx_uint64> > &locators);

            //generators:
            void set_zeros();


    };          

    typedef boost::intrusive_ptr<Storage_base> (*pStorage_init)();
    class Storage_init_interface: public Type{
        public:
            std::vector<pStorage_init> USIInit;
            Storage_init_interface();
    };
    //extern Storage_init_interface Storage_init;


    ///wrapping-up 
    class Storage{  
        private:
            //Interface:
            Storage_init_interface __SII;

        public:
            boost::intrusive_ptr<Storage_base> _impl;

            void Init(const unsigned long long &size,const unsigned int &dtype, int device=-1){
                cytnx_error_msg(dtype>=N_Type,"%s","[ERROR] invalid argument: dtype");
                this->_impl = __SII.USIInit[dtype]();
                this->_impl->Init(size,device);
            }
            Storage(const unsigned long long &size, const unsigned int &dtype, int device=-1): _impl(new Storage_base()){
                Init(size,dtype,device);
            }
            Storage(): _impl(new Storage_base()){};
            Storage(boost::intrusive_ptr<Storage_base> in_impl){
                this->_impl = in_impl;
            }
            Storage(const Storage &rhs){
                this->_impl = rhs._impl;
            }
            Storage& operator=(const Storage &rhs){
                this->_impl = rhs._impl;
                return *this;
            }

            Storage astype(const unsigned int &new_type) const{
                return this->_impl->astype(new_type);
            }

            const unsigned int &dtype() const{
                return this->_impl->dtype;
            }
            const std::string dtype_str() const{
                std::string out = this->_impl->dtype_str();
                return out;
            }

            const int &device() const{
                return this->_impl->device;
            }
    
            const std::string device_str() const{
                std::string out = this->_impl->device_str();
                return out;
            }            

            template<class T>
            T& at(const unsigned int &idx) const{
                return this->_impl->at<T>(idx);
            }
            template<class T>
            T* data() const{
                return this->_impl->data<T>();
            }
            void to_(const int &device){
                this->_impl->to_(device);
            }
            Storage to(const int &device){
                return Storage(this->_impl->to(device));
            }
            Storage clone(){
                return Storage(this->_impl->clone());
            }
            const unsigned long long &size() const{
                return this->_impl->len;
            }
            void print_info(){
                this->_impl->print_info();
            }
            void print(){
                this->_impl->print();
            }
            void set_zeros(){
                this->_impl->set_zeros();
            }
            bool operator==(const Storage &rhs);


    };
    std::ostream& operator<<(std::ostream& os, Storage &in);


}

#endif
