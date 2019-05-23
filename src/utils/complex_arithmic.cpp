#include "Type.hpp"
#include "utils/complex_arithmic.hpp"
namespace tor10{

    tor10_complex128 operator+(const tor10_complex128 &ln, const tor10_complex64 &rn)
    {
        return tor10_complex128(ln.real() + rn.real(), ln.imag() + rn.imag());
    }
    tor10_complex128 operator+(const tor10_complex128 &ln, const tor10_double &rn)
    {
        return tor10_complex128(ln.real() + rn, ln.imag() );
    }
    tor10_complex128 operator+(const tor10_complex128 &ln, const tor10_float &rn)
    {
        return tor10_complex128(ln.real() + rn, ln.imag() );
    }
    tor10_complex128 operator+(const tor10_complex128 &ln, const tor10_uint64 &rn)
    {
        return tor10_complex128(ln.real() + rn, ln.imag() );
    }
    tor10_complex128 operator+(const tor10_complex128 &ln, const tor10_uint32 &rn)
    {
        return tor10_complex128(ln.real() + rn, ln.imag() );
    }
    tor10_complex128 operator+(const tor10_complex128 &ln, const tor10_int64 &rn)
    {
        return tor10_complex128(ln.real() + rn, ln.imag() );
    }
    tor10_complex128 operator+(const tor10_complex128 &ln, const tor10_int32 &rn)
    {
        return tor10_complex128(ln.real() + rn, ln.imag() );
    }


//-----------------------------
    tor10_complex128 operator+(const tor10_complex64 &ln, const tor10_complex128 &rn)
    {
        return tor10_complex128(ln.real() + rn.real(), ln.imag()+rn.imag() );
    }

    tor10_complex64 operator+(const tor10_complex64 &ln, const tor10_double &rn)
    {
        return tor10_complex64(ln.real() + rn, ln.imag());
    }
    tor10_complex64 operator+(const tor10_complex64 &ln, const tor10_float &rn)
    {
        return tor10_complex64(ln.real() + rn, ln.imag());
    }
    tor10_complex64 operator+(const tor10_complex64 &ln, const tor10_uint64 &rn)
    {
        return tor10_complex64(ln.real() + rn, ln.imag());
    }
    tor10_complex64 operator+(const tor10_complex64 &ln, const tor10_uint32 &rn)
    {
        return tor10_complex64(ln.real() + rn, ln.imag());
    }
    tor10_complex64 operator+(const tor10_complex64 &ln, const tor10_int64 &rn)
    {
        return tor10_complex64(ln.real() + rn, ln.imag());
    }
    tor10_complex64 operator+(const tor10_complex64 &ln, const tor10_int32 &rn)
    {
        return tor10_complex64(ln.real() + rn, ln.imag());
    }

//-----------------------
    //tor10_complex128 operator+(const tor10_complex128 &rn,const tor10_complex128 &ln);
    //tor10_complex128 operator+(const tor10_complex64 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator+(const tor10_double &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(ln.real() + rn, ln.imag());
    }
    tor10_complex128 operator+(const tor10_float &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(ln.real() + rn, ln.imag());
    }
    tor10_complex128 operator+(const tor10_uint64 &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(ln.real() + rn, ln.imag());
    }
    tor10_complex128 operator+(const tor10_uint32 &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(ln.real() + rn, ln.imag());
    }
    tor10_complex128 operator+(const tor10_int64 &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(ln.real() + rn, ln.imag());
    }
    tor10_complex128 operator+(const tor10_int32 &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(ln.real() + rn, ln.imag());
    }

//----------------------

    //tor10_complex128 operator+(const tor10_complex128 &rn,const tor10_complex64 &ln);
    //tor10_complex64 operator+(const tor10_complex64 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator+(const tor10_double &rn,const tor10_complex64 &ln)
    {
        return tor10_complex64(ln.real() + rn, ln.imag());
    }
    tor10_complex64 operator+( const tor10_float &rn,const tor10_complex64 &ln)
    {
        return tor10_complex64(ln.real() + rn, ln.imag());
    }
    tor10_complex64 operator+(const tor10_uint64 &rn,const tor10_complex64 &ln)
    {
        return tor10_complex64(ln.real() + rn, ln.imag());
    }
    tor10_complex64 operator+(const tor10_uint32 &rn,const tor10_complex64 &ln)
    {
        return tor10_complex64(ln.real() + rn, ln.imag());
    }
    tor10_complex64 operator+(const tor10_int64 &rn,const tor10_complex64 &ln)    
    {
        return tor10_complex64(ln.real() + rn, ln.imag());
    }
    tor10_complex64 operator+( const tor10_int32 &rn,const tor10_complex64 &ln)
    {
        return tor10_complex64(ln.real() + rn, ln.imag());
    }

//===================================

    //tor10_complex128 operator-(const tor10_complex128 &ln, const tor10_complex128 &rn);
    tor10_complex128 operator-(const tor10_complex128 &ln, const tor10_complex64 &rn)
    { 
        return tor10_complex128(ln.real() - rn.real(), ln.imag()-rn.imag());
    }
    tor10_complex128 operator-(const tor10_complex128 &ln, const tor10_double &rn)
    { 
        return tor10_complex128(ln.real() - rn, ln.imag());
    }
    tor10_complex128 operator-(const tor10_complex128 &ln, const tor10_float &rn)
    { 
        return tor10_complex128(ln.real() - rn, ln.imag());
    }
    tor10_complex128 operator-(const tor10_complex128 &ln, const tor10_uint64 &rn)
    { 
        return tor10_complex128(ln.real() - rn, ln.imag());
    }
    tor10_complex128 operator-(const tor10_complex128 &ln, const tor10_uint32 &rn)
    { 
        return tor10_complex128(ln.real() - rn, ln.imag());
    }
    tor10_complex128 operator-(const tor10_complex128 &ln, const tor10_int64 &rn)
    { 
        return tor10_complex128(ln.real() - rn, ln.imag());
    }
    tor10_complex128 operator-(const tor10_complex128 &ln, const tor10_int32 &rn)
    { 
        return tor10_complex128(ln.real() - rn, ln.imag());
    }

    tor10_complex128 operator-(const tor10_complex64 &ln, const tor10_complex128 &rn)
    { 
        return tor10_complex128(ln.real() - rn.real(), ln.imag()-rn.imag());
    }
    //tor10_complex64 operator-(const tor10_complex64 &ln, const tor10_complex64 &rn);
    tor10_complex64 operator-(const tor10_complex64 &ln, const tor10_double &rn)
    { 
        return tor10_complex64(ln.real() - rn, ln.imag());
    }
    tor10_complex64 operator-(const tor10_complex64 &ln, const tor10_float &rn)
    { 
        return tor10_complex64(ln.real() - rn, ln.imag());
    }
    tor10_complex64 operator-(const tor10_complex64 &ln, const tor10_uint64 &rn)
    { 
        return tor10_complex64(ln.real() - rn, ln.imag());
    }
    tor10_complex64 operator-(const tor10_complex64 &ln, const tor10_uint32 &rn)
    { 
        return tor10_complex64(ln.real() - rn, ln.imag());
    }
    tor10_complex64 operator-(const tor10_complex64 &ln, const tor10_int64 &rn)
    { 
        return tor10_complex64(ln.real() - rn, ln.imag());
    }
    tor10_complex64 operator-(const tor10_complex64 &ln, const tor10_int32 &rn)
    { 
        return tor10_complex64(ln.real() - rn, ln.imag());
    }
//------------

    //tor10_complex128 operator-(const tor10_complex128 &rn,const tor10_complex128 &ln);
    //tor10_complex128 operator-(const tor10_complex64 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator-(const tor10_double &rn,const tor10_complex128 &ln)
    { 
        return tor10_complex128(rn - ln.real() , -ln.imag());
    }
    tor10_complex128 operator-(const tor10_float &rn,const tor10_complex128 &ln)
    { 
        return tor10_complex128(rn - ln.real() , -ln.imag());
    }
    tor10_complex128 operator-(const tor10_uint64 &rn,const tor10_complex128 &ln)
    { 
        return tor10_complex128(rn - ln.real() , -ln.imag());
    }
    tor10_complex128 operator-(const tor10_uint32 &rn,const tor10_complex128 &ln)
    { 
        return tor10_complex128(rn - ln.real() , -ln.imag());
    }
    tor10_complex128 operator-(const tor10_int64 &rn,const tor10_complex128 &ln)
    { 
        return tor10_complex128(rn - ln.real() , -ln.imag());
    }
    tor10_complex128 operator-(const tor10_int32 &rn,const tor10_complex128 &ln)
    { 
        return tor10_complex128(rn - ln.real() , -ln.imag());
    }

//----------------

    //tor10_complex128 operator-(const tor10_complex128 &rn,const tor10_complex64 &ln);
    //tor10_complex64 operator-(const tor10_complex64 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator-(const tor10_double &rn,const tor10_complex64 &ln)
    { 
        return tor10_complex64(rn - ln.real() , -ln.imag());
    }
    tor10_complex64 operator-( const tor10_float &rn,const tor10_complex64 &ln)
    { 
        return tor10_complex64(rn - ln.real() , -ln.imag());
    }
    tor10_complex64 operator-(const tor10_uint64 &rn,const tor10_complex64 &ln)
    { 
        return tor10_complex64(rn - ln.real() , -ln.imag());
    }
    tor10_complex64 operator-(const tor10_uint32 &rn,const tor10_complex64 &ln)
    { 
        return tor10_complex64(rn - ln.real() , -ln.imag());
    }
    tor10_complex64 operator-(const tor10_int64 &rn,const tor10_complex64 &ln)
    { 
        return tor10_complex64(rn - ln.real() , -ln.imag());
    }
    tor10_complex64 operator-( const tor10_int32 &rn,const tor10_complex64 &ln)
    { 
        return tor10_complex64(rn - ln.real() , -ln.imag());
    }


//=============================


    //tor10_complex128 operator*(const tor10_complex128 &ln, const tor10_complex128 &rn);
    tor10_complex128 operator*(const tor10_complex128 &ln, const tor10_complex64 &rn)
    {
        return ln*tor10_complex128(rn.real(),rn.imag());
    }
    tor10_complex128 operator*(const tor10_complex128 &ln, const tor10_double &rn)
    {
        return tor10_complex128(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex128 operator*(const tor10_complex128 &ln, const tor10_float &rn)
    {
        return tor10_complex128(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex128 operator*(const tor10_complex128 &ln, const tor10_uint64 &rn)
    {
        return tor10_complex128(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex128 operator*(const tor10_complex128 &ln, const tor10_uint32 &rn)
    {
        return tor10_complex128(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex128 operator*(const tor10_complex128 &ln, const tor10_int64 &rn)
    {
        return tor10_complex128(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex128 operator*(const tor10_complex128 &ln, const tor10_int32 &rn)
    {
        return tor10_complex128(ln.real()*rn,ln.imag()*rn);
    }

    tor10_complex128 operator*(const tor10_complex64 &ln, const tor10_complex128 &rn)
    {
        return rn*tor10_complex128(ln.real(),ln.imag());
    }
    //tor10_complex64 operator*(const tor10_complex64 &ln, const tor10_complex64 &rn);
    tor10_complex64 operator*(const tor10_complex64 &ln, const tor10_double &rn)
    {
        return tor10_complex64(ln.real()*rn,ln.imag()*rn);
    }

    tor10_complex64 operator*(const tor10_complex64 &ln, const tor10_float &rn)
    {
        return tor10_complex64(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex64 operator*(const tor10_complex64 &ln, const tor10_uint64 &rn)
    {
        return tor10_complex64(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex64 operator*(const tor10_complex64 &ln, const tor10_uint32 &rn)
    {
        return tor10_complex64(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex64 operator*(const tor10_complex64 &ln, const tor10_int64 &rn)
    {
        return tor10_complex64(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex64 operator*(const tor10_complex64 &ln, const tor10_int32 &rn)
    {
        return tor10_complex64(ln.real()*rn,ln.imag()*rn);
    }

    //tor10_complex128 operator*(const tor10_complex128 &rn,const tor10_complex128 &ln);
    //tor10_complex128 operator*(const tor10_complex64 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator*(const tor10_double &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex128 operator*(const tor10_float &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex128 operator*(const tor10_uint64 &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex128 operator*(const tor10_uint32 &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex128 operator*(const tor10_int64 &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex128 operator*(const tor10_int32 &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(ln.real()*rn,ln.imag()*rn);
    }

    //tor10_complex128 operator*(const tor10_complex128 &rn,const tor10_complex64 &ln);
    //tor10_complex64 operator*(const tor10_complex64 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator*(const tor10_double &rn,const tor10_complex64 &ln)
    {
        return tor10_complex64(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex64 operator*( const tor10_float &rn,const tor10_complex64 &ln)
    {
        return tor10_complex64(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex64 operator*(const tor10_uint64 &rn,const tor10_complex64 &ln)
    {
        return tor10_complex64(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex64 operator*(const tor10_uint32 &rn,const tor10_complex64 &ln)
    {
        return tor10_complex64(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex64 operator*(const tor10_int64 &rn,const tor10_complex64 &ln)
    {
        return tor10_complex64(ln.real()*rn,ln.imag()*rn);
    }
    tor10_complex64 operator*( const tor10_int32 &rn,const tor10_complex64 &ln)
    {
        return tor10_complex64(ln.real()*rn,ln.imag()*rn);
    }

//-------------------------------

    //tor10_complex128 operator/(const tor10_complex128 &ln, const tor10_complex128 &rn);
    tor10_complex128 operator/(const tor10_complex128 &ln, const tor10_complex64 &rn)
    {
        return ln/tor10_complex128(rn);
    }
    tor10_complex128 operator/(const tor10_complex128 &ln, const tor10_double &rn)
    {
        return tor10_complex128(ln.real()/rn,ln.imag()/rn);
    }
    tor10_complex128 operator/(const tor10_complex128 &ln, const tor10_float &rn)
    {
        return tor10_complex128(ln.real()/rn,ln.imag()/rn);
    }
    tor10_complex128 operator/(const tor10_complex128 &ln, const tor10_uint64 &rn)
    {
        return tor10_complex128(ln.real()/rn,ln.imag()/rn);
    }
    tor10_complex128 operator/(const tor10_complex128 &ln, const tor10_uint32 &rn)
    {
        return tor10_complex128(ln.real()/rn,ln.imag()/rn);
    }
    tor10_complex128 operator/(const tor10_complex128 &ln, const tor10_int64 &rn)
    {
        return tor10_complex128(ln.real()/rn,ln.imag()/rn);
    }
    tor10_complex128 operator/(const tor10_complex128 &ln, const tor10_int32 &rn)
    {
        return tor10_complex128(ln.real()/rn,ln.imag()/rn);
    }

    tor10_complex128 operator/(const tor10_complex64 &ln, const tor10_complex128 &rn)
    {
        return tor10_complex128(ln)/rn;
    }
    //tor10_complex64 operator/(const tor10_complex64 &ln, const tor10_complex64 &rn);
    tor10_complex64 operator/(const tor10_complex64 &ln, const tor10_double &rn)
    {
        return tor10_complex64(ln.real()/rn,ln.imag()/rn);
    }
    tor10_complex64 operator/(const tor10_complex64 &ln, const tor10_float &rn)
    {
        return tor10_complex64(ln.real()/rn,ln.imag()/rn);
    }
    tor10_complex64 operator/(const tor10_complex64 &ln, const tor10_uint64 &rn)
    {
        return tor10_complex64(ln.real()/rn,ln.imag()/rn);
    }
    tor10_complex64 operator/(const tor10_complex64 &ln, const tor10_uint32 &rn)
    {
        return tor10_complex64(ln.real()/rn,ln.imag()/rn);
    }
    tor10_complex64 operator/(const tor10_complex64 &ln, const tor10_int64 &rn)
    {
        return tor10_complex64(ln.real()/rn,ln.imag()/rn);
    }
    tor10_complex64 operator/(const tor10_complex64 &ln, const tor10_int32 &rn)
    {
        return tor10_complex64(ln.real()/rn,ln.imag()/rn);
    }

    //tor10_complex128 operator/(const tor10_complex128 &rn,const tor10_complex128 &ln);
    //tor10_complex128 operator/(const tor10_complex64 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator/(const tor10_double &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(rn,0)/ln;
    }
    tor10_complex128 operator/(const tor10_float &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(rn,0)/ln;
    }
    tor10_complex128 operator/(const tor10_uint64 &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(rn,0)/ln;
    }
    tor10_complex128 operator/(const tor10_uint32 &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(rn,0)/ln;
    }
    tor10_complex128 operator/(const tor10_int64 &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(rn,0)/ln;
    }
    tor10_complex128 operator/(const tor10_int32 &rn,const tor10_complex128 &ln)
    {
        return tor10_complex128(rn,0)/ln;
    }

    //tor10_complex128 operator/(const tor10_complex128 &rn,const tor10_complex64 &ln);
    //tor10_complex64 operator/(const tor10_complex64 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator/(const tor10_double &rn,const tor10_complex64 &ln)
    {
        return tor10_complex64(rn,0)/ln;
    }
    tor10_complex64 operator/( const tor10_float &rn,const tor10_complex64 &ln)
    {
        return tor10_complex64(rn,0)/ln;
    }
    tor10_complex64 operator/(const tor10_uint64 &rn,const tor10_complex64 &ln)
    {
        return tor10_complex64(rn,0)/ln;
    }
    tor10_complex64 operator/(const tor10_uint32 &rn,const tor10_complex64 &ln)
    {
        return tor10_complex64(rn,0)/ln;
    }
    tor10_complex64 operator/(const tor10_int64 &rn,const tor10_complex64 &ln)
    {
        return tor10_complex64(rn,0)/ln;
    }
    tor10_complex64 operator/( const tor10_int32 &rn,const tor10_complex64 &ln)
    {
        return tor10_complex64(rn,0)/ln;
    }


}//namespace tor10

