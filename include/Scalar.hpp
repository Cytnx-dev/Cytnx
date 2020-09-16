#ifndef _H_Scalar_
#define _H_Scalar_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "intrusive_ptr_base.hpp"
#include <vector>
#include <initializer_list>
#include <string>
#include <iostream>

namespace cytnx{


    ///@cond
    // real implementation
    class Scalar_base{
        private:
            
        public:
            int _dtype;

            //Scalar_base(const Scalar_base &rhs);
            //Scalar_base& operator=(const Scalar_base &rhs);   
            Scalar_base(): _dtype(Type.Void){};

            virtual cytnx_float  to_cytnx_float() const{cytnx_error_msg(true,"[ERROR] Void Type Scalar cannot cast to anytype!!%s","\n"); return 0;};
            virtual cytnx_double to_cytnx_double() const{cytnx_error_msg(true,"[ERROR] Void Type Scalar cannot cast to anytype!!%s","\n"); return 0;};
            virtual cytnx_complex64 to_cytnx_complex64() const{cytnx_error_msg(true,"[ERROR] Void Type Scalar cannot cast to anytype!!%s","\n");return cytnx_complex64(0,0);};
            virtual cytnx_complex128 to_cytnx_complex128() const{cytnx_error_msg(true,"[ERROR] Void Type Scalar cannot cast to anytype!!%s","\n");return cytnx_complex128(0,0);};
            virtual cytnx_int64 to_cytnx_int64() const{cytnx_error_msg(true,"[ERROR] Void Type Scalar cannot cast to anytype!!%s","\n");return 0;};
            virtual cytnx_uint64 to_cytnx_uint64() const{cytnx_error_msg(true,"[ERROR] Void Type Scalar cannot cast to anytype!!%s","\n");return 0;};
            virtual cytnx_int32 to_cytnx_int32() const{cytnx_error_msg(true,"[ERROR] Void Type Scalar cannot cast to anytype!!%s","\n");return 0;};
            virtual cytnx_uint32 to_cytnx_uint32() const{cytnx_error_msg(true,"[ERROR] Void Type Scalar cannot cast to anytype!!%s","\n");return 0;};
            virtual cytnx_int16 to_cytnx_int16() const{cytnx_error_msg(true,"[ERROR] Void Type Scalar cannot cast to anytype!!%s","\n");return 0;};
            virtual cytnx_uint16 to_cytnx_uint16() const{cytnx_error_msg(true,"[ERROR] Void Type Scalar cannot cast to anytype!!%s","\n");return 0;};
            virtual cytnx_bool to_cytnx_bool() const{cytnx_error_msg(true,"[ERROR] Void Type Scalar cannot cast to anytype!!%s","\n");return 0;}
            virtual void print(std::ostream& os) const{};
            virtual Scalar_base* copy() const{
                Scalar_base *tmp = new Scalar_base();
                return tmp;
            };
           

    };
        

    class ComplexDoubleScalar: public Scalar_base{

        public:
            cytnx_complex128 _elem;

            ComplexDoubleScalar(): _elem(0){this->_dtype = Type.ComplexDouble;};
            ComplexDoubleScalar(const cytnx_complex128 &in):_elem(0){
                this->_dtype = Type.ComplexDouble;
                this->_elem = in;
            }
            
            cytnx_float  to_cytnx_float()          const{cytnx_error_msg(true,"[ERROR] cannot cast complex128 to real%s","\n");return 0;};
            cytnx_double to_cytnx_double()         const{cytnx_error_msg(true,"[ERROR] cannot cast complex128 to real%s","\n");return 0;};
            cytnx_complex64 to_cytnx_complex64()   const{return cytnx_complex64(this->_elem);};
            cytnx_complex128 to_cytnx_complex128() const{return this->_elem;};
            cytnx_int64 to_cytnx_int64()           const{cytnx_error_msg(true,"[ERROR] cannot cast complex128 to real%s","\n");return 0;};
            cytnx_uint64 to_cytnx_uint64()         const{cytnx_error_msg(true,"[ERROR] cannot cast complex128 to real%s","\n");return 0;};
            cytnx_int32 to_cytnx_int32()           const{cytnx_error_msg(true,"[ERROR] cannot cast complex128 to real%s","\n");return 0;};
            cytnx_uint32 to_cytnx_uint32()         const{cytnx_error_msg(true,"[ERROR] cannot cast complex128 to real%s","\n");return 0;};
            cytnx_int16 to_cytnx_int16()           const{cytnx_error_msg(true,"[ERROR] cannot cast complex128 to real%s","\n");return 0;};
            cytnx_uint16 to_cytnx_uint16()         const{cytnx_error_msg(true,"[ERROR] cannot cast complex128 to real%s","\n");return 0;};
            cytnx_bool to_cytnx_bool()             const{cytnx_error_msg(true,"[ERROR] cannot cast complex128 to real%s","\n");return 0;};
            Scalar_base* copy() const{
                ComplexDoubleScalar *tmp = new ComplexDoubleScalar(this->_elem); 
                return tmp;
            };
            void print(std::ostream& os) const{
                os << this->_elem << std::endl;
            };
    }; 

    class ComplexFloatScalar: public Scalar_base{

        public:
            cytnx_complex64 _elem;

            ComplexFloatScalar(): _elem(0){this->_dtype = Type.ComplexFloat;};
            ComplexFloatScalar(const cytnx_complex64 &in): _elem(0){
                this->_dtype = Type.ComplexFloat;
                this->_elem = in;
            }
            
            cytnx_float  to_cytnx_float()          const{cytnx_error_msg(true,"[ERROR] cannot cast complex64 to real%s","\n");return 0;};
            cytnx_double to_cytnx_double()         const{cytnx_error_msg(true,"[ERROR] cannot cast complex64 to real%s","\n");return 0;};
            cytnx_complex64 to_cytnx_complex64()   const{return this->_elem;};
            cytnx_complex128 to_cytnx_complex128() const{return cytnx_complex128(this->_elem);};
            cytnx_int64 to_cytnx_int64()           const{cytnx_error_msg(true,"[ERROR] cannot cast complex64 to real%s","\n");return 0;};
            cytnx_uint64 to_cytnx_uint64()         const{cytnx_error_msg(true,"[ERROR] cannot cast complex64 to real%s","\n");return 0;};
            cytnx_int32 to_cytnx_int32()           const{cytnx_error_msg(true,"[ERROR] cannot cast complex64 to real%s","\n");return 0;};
            cytnx_uint32 to_cytnx_uint32()         const{cytnx_error_msg(true,"[ERROR] cannot cast complex64 to real%s","\n");return 0;};
            cytnx_int16 to_cytnx_int16()           const{cytnx_error_msg(true,"[ERROR] cannot cast complex64 to real%s","\n");return 0;};
            cytnx_uint16 to_cytnx_uint16()         const{cytnx_error_msg(true,"[ERROR] cannot cast complex64 to real%s","\n");return 0;};
            cytnx_bool to_cytnx_bool()             const{cytnx_error_msg(true,"[ERROR] cannot cast complex64 to real%s","\n");return 0;};
            Scalar_base* copy() const{
                ComplexFloatScalar *tmp = new ComplexFloatScalar(this->_elem); 
                return tmp;
            };
            void print(std::ostream& os) const{
                os << this->_elem << std::endl;
            };
    }; 

    class DoubleScalar: public Scalar_base{

        public:
            cytnx_double _elem;

            DoubleScalar(): _elem(0){this->_dtype = Type.Double;};
            DoubleScalar(const cytnx_double &in): _elem(0){
                this->_dtype = Type.Double;
                this->_elem = in;
            }
            
            cytnx_float  to_cytnx_float()          const{return this->_elem;};
            cytnx_double to_cytnx_double()         const{return this->_elem;};
            cytnx_complex64 to_cytnx_complex64()   const{return this->_elem;};
            cytnx_complex128 to_cytnx_complex128() const{return this->_elem;};
            cytnx_int64 to_cytnx_int64()           const{return this->_elem;};
            cytnx_uint64 to_cytnx_uint64()         const{return this->_elem;};
            cytnx_int32 to_cytnx_int32()           const{return this->_elem;};
            cytnx_uint32 to_cytnx_uint32()         const{return this->_elem;};
            cytnx_int16 to_cytnx_int16()           const{return this->_elem;};
            cytnx_uint16 to_cytnx_uint16()         const{return this->_elem;};
            cytnx_bool to_cytnx_bool()             const{return this->_elem;};
            Scalar_base* copy() const{
                DoubleScalar *tmp = new DoubleScalar(this->_elem); 
                return tmp;
            };
            void print(std::ostream& os) const{
                os << this->_elem << std::endl;
            };
    }; 

    class FloatScalar: public Scalar_base{

        public:
            cytnx_float _elem;

            FloatScalar(): _elem(0){this->_dtype = Type.Float;};
            FloatScalar(const cytnx_float &in): _elem(0){
                this->_dtype = Type.Float;
                this->_elem = in;
            }
            
            cytnx_float  to_cytnx_float()          const{return this->_elem;};
            cytnx_double to_cytnx_double()         const{return this->_elem;};
            cytnx_complex64 to_cytnx_complex64()   const{return this->_elem;};
            cytnx_complex128 to_cytnx_complex128() const{return this->_elem;};
            cytnx_int64 to_cytnx_int64()           const{return this->_elem;};
            cytnx_uint64 to_cytnx_uint64()         const{return this->_elem;};
            cytnx_int32 to_cytnx_int32()           const{return this->_elem;};
            cytnx_uint32 to_cytnx_uint32()         const{return this->_elem;};
            cytnx_int16 to_cytnx_int16()           const{return this->_elem;};
            cytnx_uint16 to_cytnx_uint16()         const{return this->_elem;};
            cytnx_bool to_cytnx_bool()             const{return this->_elem;};
            Scalar_base* copy() const{
                FloatScalar *tmp = new FloatScalar(this->_elem); 
                return tmp;
            };
            void print(std::ostream& os) const{
                os << this->_elem << std::endl;
            };
    }; 

    class Int64Scalar: public Scalar_base{

        public:
            cytnx_int64 _elem;

            Int64Scalar(): _elem(0){this->_dtype = Type.Int64;};
            Int64Scalar(const cytnx_int64 &in): _elem(0){
                this->_dtype = Type.Int64;
                this->_elem = in;
            }
            
            cytnx_float  to_cytnx_float()          const{return this->_elem;};
            cytnx_double to_cytnx_double()         const{return this->_elem;};
            cytnx_complex64 to_cytnx_complex64()   const{return this->_elem;};
            cytnx_complex128 to_cytnx_complex128() const{return this->_elem;};
            cytnx_int64 to_cytnx_int64()           const{return this->_elem;};
            cytnx_uint64 to_cytnx_uint64()         const{return this->_elem;};
            cytnx_int32 to_cytnx_int32()           const{return this->_elem;};
            cytnx_uint32 to_cytnx_uint32()         const{return this->_elem;};
            cytnx_int16 to_cytnx_int16()           const{return this->_elem;};
            cytnx_uint16 to_cytnx_uint16()         const{return this->_elem;};
            cytnx_bool to_cytnx_bool()             const{return this->_elem;};
            Scalar_base* copy() const{
                Int64Scalar *tmp = new Int64Scalar(this->_elem); 
                return tmp;
            };
            void print(std::ostream& os) const{
                os << this->_elem << std::endl;
            };
    }; 
    class Uint64Scalar: public Scalar_base{

        public:
            cytnx_uint64 _elem;

            Uint64Scalar(): _elem(0){this->_dtype = Type.Uint64;};
            Uint64Scalar(const cytnx_uint64 &in): _elem(0){
                this->_dtype = Type.Uint64;
                this->_elem = in;
            }
            
            cytnx_float  to_cytnx_float()          const{return this->_elem;};
            cytnx_double to_cytnx_double()         const{return this->_elem;};
            cytnx_complex64 to_cytnx_complex64()   const{return this->_elem;};
            cytnx_complex128 to_cytnx_complex128() const{return this->_elem;};
            cytnx_int64 to_cytnx_int64()           const{return this->_elem;};
            cytnx_uint64 to_cytnx_uint64()         const{return this->_elem;};
            cytnx_int32 to_cytnx_int32()           const{return this->_elem;};
            cytnx_uint32 to_cytnx_uint32()         const{return this->_elem;};
            cytnx_int16 to_cytnx_int16()           const{return this->_elem;};
            cytnx_uint16 to_cytnx_uint16()         const{return this->_elem;};
            cytnx_bool to_cytnx_bool()             const{return this->_elem;};
            Scalar_base* copy() const{
                Uint64Scalar *tmp = new Uint64Scalar(this->_elem); 
                return tmp;
            };
            void print(std::ostream& os) const{
                os << this->_elem << std::endl;
            };
    }; 
    class Int32Scalar: public Scalar_base{

        public:
            cytnx_int32 _elem;

            Int32Scalar(): _elem(0){this->_dtype = Type.Int32;};
            Int32Scalar(const cytnx_int32 &in): _elem(0){
                this->_dtype = Type.Int32;
                this->_elem = in;
            }
            
            cytnx_float  to_cytnx_float()          const{return this->_elem;};
            cytnx_double to_cytnx_double()         const{return this->_elem;};
            cytnx_complex64 to_cytnx_complex64()   const{return this->_elem;};
            cytnx_complex128 to_cytnx_complex128() const{return this->_elem;};
            cytnx_int64 to_cytnx_int64()           const{return this->_elem;};
            cytnx_uint64 to_cytnx_uint64()         const{return this->_elem;};
            cytnx_int32 to_cytnx_int32()           const{return this->_elem;};
            cytnx_uint32 to_cytnx_uint32()         const{return this->_elem;};
            cytnx_int16 to_cytnx_int16()           const{return this->_elem;};
            cytnx_uint16 to_cytnx_uint16()         const{return this->_elem;};
            cytnx_bool to_cytnx_bool()             const{return this->_elem;};
            Scalar_base* copy() const{
                Int32Scalar *tmp = new Int32Scalar(this->_elem); 
                return tmp;
            };
            void print(std::ostream& os) const{
                os << this->_elem << std::endl;
            };
    }; 
    class Uint32Scalar: public Scalar_base{

        public:
            cytnx_uint32 _elem;

            Uint32Scalar(): _elem(0){this->_dtype = Type.Uint32;};
            Uint32Scalar(const cytnx_uint32 &in): _elem(0){
                this->_dtype = Type.Uint32;
                this->_elem = in;
            }
            
            cytnx_float  to_cytnx_float()          const{return this->_elem;};
            cytnx_double to_cytnx_double()         const{return this->_elem;};
            cytnx_complex64 to_cytnx_complex64()   const{return this->_elem;};
            cytnx_complex128 to_cytnx_complex128() const{return this->_elem;};
            cytnx_int64 to_cytnx_int64()           const{return this->_elem;};
            cytnx_uint64 to_cytnx_uint64()         const{return this->_elem;};
            cytnx_int32 to_cytnx_int32()           const{return this->_elem;};
            cytnx_uint32 to_cytnx_uint32()         const{return this->_elem;};
            cytnx_int16 to_cytnx_int16()           const{return this->_elem;};
            cytnx_uint16 to_cytnx_uint16()         const{return this->_elem;};
            cytnx_bool to_cytnx_bool()             const{return this->_elem;};
            Scalar_base* copy() const{
                Uint32Scalar *tmp = new Uint32Scalar(this->_elem); 
                return tmp;
            };
            void print(std::ostream& os) const{
                os << this->_elem << std::endl;
            };
    }; 
    class Int16Scalar: public Scalar_base{

        public:
            cytnx_int16 _elem;

            Int16Scalar(): _elem(0){this->_dtype = Type.Int16;};
            Int16Scalar(const cytnx_int16 &in): _elem(0){
                this->_dtype = Type.Int16;
                this->_elem = in;
            }
            
            cytnx_float  to_cytnx_float()          const{return this->_elem;};
            cytnx_double to_cytnx_double()         const{return this->_elem;};
            cytnx_complex64 to_cytnx_complex64()   const{return this->_elem;};
            cytnx_complex128 to_cytnx_complex128() const{return this->_elem;};
            cytnx_int64 to_cytnx_int64()           const{return this->_elem;};
            cytnx_uint64 to_cytnx_uint64()         const{return this->_elem;};
            cytnx_int32 to_cytnx_int32()           const{return this->_elem;};
            cytnx_uint32 to_cytnx_uint32()         const{return this->_elem;};
            cytnx_int16 to_cytnx_int16()           const{return this->_elem;};
            cytnx_uint16 to_cytnx_uint16()         const{return this->_elem;};
            cytnx_bool to_cytnx_bool()             const{return this->_elem;};
            Scalar_base* copy() const{
                Int16Scalar *tmp = new Int16Scalar(this->_elem); 
                return tmp;
            };
            void print(std::ostream& os) const{
                os << this->_elem << std::endl;
            };
    };
    class Uint16Scalar: public Scalar_base{

        public:
            cytnx_uint16 _elem;

            Uint16Scalar(): _elem(0){this->_dtype = Type.Uint16;};
            Uint16Scalar(const cytnx_uint16 &in): _elem(0){
                this->_dtype = Type.Uint16;
                this->_elem = in;
            }
            
            cytnx_float  to_cytnx_float()          const{return this->_elem;};
            cytnx_double to_cytnx_double()         const{return this->_elem;};
            cytnx_complex64 to_cytnx_complex64()   const{return this->_elem;};
            cytnx_complex128 to_cytnx_complex128() const{return this->_elem;};
            cytnx_int64 to_cytnx_int64()           const{return this->_elem;};
            cytnx_uint64 to_cytnx_uint64()         const{return this->_elem;};
            cytnx_int32 to_cytnx_int32()           const{return this->_elem;};
            cytnx_uint32 to_cytnx_uint32()         const{return this->_elem;};
            cytnx_int16 to_cytnx_int16()           const{return this->_elem;};
            cytnx_uint16 to_cytnx_uint16()         const{return this->_elem;};
            cytnx_bool to_cytnx_bool()             const{return this->_elem;};
            Scalar_base* copy() const{
                Uint16Scalar *tmp = new Uint16Scalar(this->_elem); 
                return tmp;
            };
            void print(std::ostream& os) const{
                os << this->_elem << std::endl;
            };
    }; 
    class BoolScalar: public Scalar_base{

        public:
            cytnx_bool _elem;

            BoolScalar(): _elem(0){this->_dtype = Type.Bool;};
            BoolScalar(const cytnx_bool &in): _elem(0){
                this->_dtype = Type.Bool;
                this->_elem = in;
            }
            
            cytnx_float  to_cytnx_float()          const{return this->_elem;};
            cytnx_double to_cytnx_double()         const{return this->_elem;};
            cytnx_complex64 to_cytnx_complex64()   const{return this->_elem;};
            cytnx_complex128 to_cytnx_complex128() const{return this->_elem;};
            cytnx_int64 to_cytnx_int64()           const{return this->_elem;};
            cytnx_uint64 to_cytnx_uint64()         const{return this->_elem;};
            cytnx_int32 to_cytnx_int32()           const{return this->_elem;};
            cytnx_uint32 to_cytnx_uint32()         const{return this->_elem;};
            cytnx_int16 to_cytnx_int16()           const{return this->_elem;};
            cytnx_uint16 to_cytnx_uint16()         const{return this->_elem;};
            cytnx_bool to_cytnx_bool()             const{return this->_elem;};
            Scalar_base* copy() const{
                BoolScalar *tmp = new BoolScalar(this->_elem); 
                return tmp;
            };
            void print(std::ostream& os) const{
                os << this->_elem << std::endl;
            };
    }; 

    ///@endcond



    class Scalar{
        public:
            Scalar_base* _impl;

            Scalar(): _impl(new Scalar_base()){};

            // init!!        
            template<class T>
            Scalar(const T &in): _impl(new Scalar_base()){
                this->Init_by_number(in);
            }

            //specialization of init: 
            ///@cond
            void Init_by_number(const cytnx_complex128 &in){
                if(this->_impl!=nullptr) delete this->_impl;
                this->_impl = new ComplexDoubleScalar(in);
            };
            void Init_by_number(const cytnx_complex64 &in){
                if(this->_impl!=nullptr) delete this->_impl;
                this->_impl = new ComplexFloatScalar(in);
            };
            void Init_by_number(const cytnx_double &in){
                if(this->_impl!=nullptr) delete this->_impl;
                this->_impl = new DoubleScalar(in);
            }
            void Init_by_number(const cytnx_float &in){
                if(this->_impl!=nullptr) delete this->_impl;
                this->_impl = new FloatScalar(in);
            }
            void Init_by_number(const cytnx_int64 &in){
                if(this->_impl!=nullptr) delete this->_impl;
                this->_impl = new Int64Scalar(in);
            }
            void Init_by_number(const cytnx_uint64 &in){
                if(this->_impl!=nullptr) delete this->_impl;
                this->_impl = new Uint64Scalar(in);
            }
            void Init_by_number(const cytnx_int32 &in){
                if(this->_impl!=nullptr) delete this->_impl;
                this->_impl = new Int32Scalar(in);
            }
            void Init_by_number(const cytnx_uint32 &in){
                if(this->_impl!=nullptr) delete this->_impl;
                this->_impl = new Uint32Scalar(in);
            }
            void Init_by_number(const cytnx_int16 &in){
                if(this->_impl!=nullptr) delete this->_impl;
                this->_impl = new Int16Scalar(in);
            }
            void Init_by_number(const cytnx_uint16 &in){
                if(this->_impl!=nullptr) delete this->_impl;
                this->_impl = new Uint16Scalar(in);
            }
            void Init_by_number(const cytnx_bool &in){
                if(this->_impl!=nullptr) delete this->_impl;
                this->_impl = new BoolScalar(in);
            }
            /// @endcond
            

            // copy constructor [Scalar]:
            Scalar(const Scalar &rhs){
                if(this->_impl!=nullptr)
                    delete this->_impl;
                
                this->_impl = rhs._impl->copy();
            }
             
            // copy assignment [Scalar]:
            Scalar& operator=(const Scalar &rhs){
                if(this->_impl!=nullptr)
                    delete this->_impl;

                this->_impl = rhs._impl->copy();
                return *this;
            };
                
            // copy assignment [Number]:
            template<class T>
            Scalar& operator=(const T &rhs){
                this->Init_by_number(rhs);
                return *this;
            }

            int dtype() const{
                return this->_impl->_dtype;
            }            

            // print()
            void print() const{
                std::cout << std::string("Scalar dtype: [") << Type.getname(this->_impl->_dtype) << std::string("]") << std::endl;
                this->_impl->print(std::cout);
            }

            
            operator cytnx_double() const{
                return this->_impl->to_cytnx_double();
            }
            operator cytnx_float() const{
                return this->_impl->to_cytnx_float();
            }
            operator cytnx_uint64() const{
                return this->_impl->to_cytnx_uint64();
            }
            operator cytnx_int64() const{
                return this->_impl->to_cytnx_int64();
            }
            operator cytnx_uint32() const{
                return this->_impl->to_cytnx_uint32();
            }
            operator cytnx_int32() const{
                return this->_impl->to_cytnx_int32();
            }
            operator cytnx_uint16() const{
                return this->_impl->to_cytnx_uint16();
            }
            operator cytnx_int16() const{
                return this->_impl->to_cytnx_int16();
            }
            operator cytnx_bool() const{
                return this->_impl->to_cytnx_bool();
            }
            ~Scalar(){
                if(this->_impl!=nullptr)
                    delete this->_impl;
            };
            

    };

    cytnx_complex128 complex128(const Scalar &in);

    cytnx_complex64 complex64(const Scalar &in);

    std::ostream& operator<<(std::ostream& os, const Scalar &in);

}

#endif
