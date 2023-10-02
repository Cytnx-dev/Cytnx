#ifndef _H_Scalar_
#define _H_Scalar_
#ifdef BACKEND_TORCH
  #include <torch/torch.h>

namespace cytnx {
  class Scalar : public c10::Scalar {
   public:
    ///@cond
    struct Sproxy {
      boost::intrusive_ptr<Storage_base> _insimpl;
      cytnx_uint64 _loc;
      Sproxy() {}
      Sproxy(boost::intrusive_ptr<Storage_base> _ptr, const cytnx_uint64 &idx)
          : _insimpl(_ptr), _loc(idx) {}

      Sproxy(const Sproxy &rhs) {
        this->_insimpl = rhs._insimpl;
        this->_loc = rhs._loc;
      }

      // When used to set elems:
      Sproxy &operator=(const Scalar &rc);
      Sproxy &operator=(const cytnx_complex128 &rc);
      Sproxy &operator=(const cytnx_complex64 &rc);
      Sproxy &operator=(const cytnx_double &rc);
      Sproxy &operator=(const cytnx_float &rc);
      Sproxy &operator=(const cytnx_uint64 &rc);
      Sproxy &operator=(const cytnx_int64 &rc);
      Sproxy &operator=(const cytnx_uint32 &rc);
      Sproxy &operator=(const cytnx_int32 &rc);
      Sproxy &operator=(const cytnx_uint16 &rc);
      Sproxy &operator=(const cytnx_int16 &rc);
      Sproxy &operator=(const cytnx_bool &rc);

      Sproxy &operator=(const Sproxy &rc);

      Sproxy copy() const {
        Sproxy out = *this;
        return out;
      }

      Scalar real();
      Scalar imag();
      bool exists() const;

      // When used to get elements:
      // operator Scalar() const;
    };

    Scalar_base *_impl;
    ///@endcond

    /// @brief default constructor
    Scalar() : _impl(new Scalar_base()){};

    // init!!
    /// @brief init a Scalar with a cytnx::cytnx_complex128
    Scalar(const cytnx_complex128 &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_complex64
    Scalar(const cytnx_complex64 &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_double
    Scalar(const cytnx_double &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_float
    Scalar(const cytnx_float &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_uint64
    Scalar(const cytnx_uint64 &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_int64
    Scalar(const cytnx_int64 &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_uint32
    Scalar(const cytnx_uint32 &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_int32
    Scalar(const cytnx_int32 &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_uint16
    Scalar(const cytnx_uint16 &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_int16
    Scalar(const cytnx_int16 &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_bool
    Scalar(const cytnx_bool &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /**
     * @brief Get the max value of the Scalar with the given \p dtype.
     * @details This function is used to get the max value of the Scalar with the given \p dtype.
     * That is, for example, if you want to get the max value of a Scalar with
     * \p dtype = cytnx::Type.Int16, then you will get the max value of a 16-bit integer 32767.
     * @param[in] dtype The data type of the Scalar.
     * @return The max value of the Scalar with the given \p dtype.
     */
    static Scalar maxval(const unsigned int &dtype) {
      Scalar out(0, dtype);
      out._impl->set_maxval();
      return out;
    }

    /**
     * @brief Get the min value of the Scalar with the given \p dtype.
     * @details This function is used to get the min value of the Scalar with the given \p dtype.
     * That is, for example, if you want to get the min value of a Scalar with
     * \p dtype = cytnx::Type.Int16, then you will get the min value of a 16-bit integer -32768.
     * @param[in] dtype The data type of the Scalar.
     * @return The min value of the Scalar with the given \p dtype.
     */
    static Scalar minval(const unsigned int &dtype) {
      Scalar out(0, dtype);
      out._impl->set_minval();
      return out;
    }

    /**
     * @brief The constructor of the Scalar class.
     * @details This constructor is used to init a Scalar with a given template value
     *  \p in and \p dtype (see cytnx::Type).
     * @param[in] in The value of the Scalar.
     * @param[in] dtype The data type of the Scalar.
     * @return A Scalar object.
     * @note The \p dtype can be any of the cytnx::Type.
     */
    template <class T>
    Scalar(const T &in, const unsigned int &dtype) : _impl(new Scalar_base()) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = __ScII.UScIInit[dtype]();
      this->_impl->assign_selftype(in);
    };

    /// @cond
    // move sproxy when use to get elements here.
    Scalar(const Sproxy &prox);

    //[Internal!!]
    Scalar(Scalar_base *in) { this->_impl = in; }
    /// @endcond

    // specialization of init:
    ///@cond
    void Init_by_number(const cytnx_complex128 &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new ComplexDoubleScalar(in);
    };
    void Init_by_number(const cytnx_complex64 &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new ComplexFloatScalar(in);
    };
    void Init_by_number(const cytnx_double &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new DoubleScalar(in);
    }
    void Init_by_number(const cytnx_float &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new FloatScalar(in);
    }
    void Init_by_number(const cytnx_int64 &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new Int64Scalar(in);
    }
    void Init_by_number(const cytnx_uint64 &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new Uint64Scalar(in);
    }
    void Init_by_number(const cytnx_int32 &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new Int32Scalar(in);
    }
    void Init_by_number(const cytnx_uint32 &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new Uint32Scalar(in);
    }
    void Init_by_number(const cytnx_int16 &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new Int16Scalar(in);
    }
    void Init_by_number(const cytnx_uint16 &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new Uint16Scalar(in);
    }
    void Init_by_number(const cytnx_bool &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new BoolScalar(in);
    }

    // The copy constructor
    Scalar(const Scalar &rhs) : _impl(new Scalar_base()) {
      if (this->_impl != nullptr) delete this->_impl;

      this->_impl = rhs._impl->copy();
    }
    /// @endcond

    /// @brief The copy assignment of the Scalar class.
    Scalar &operator=(const Scalar &rhs) {
      if (this->_impl != nullptr) delete this->_impl;

      this->_impl = rhs._impl->copy();
      return *this;
    };

    // copy assignment [Number]:
    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_complex128 \p rhs.
     */
    Scalar &operator=(const cytnx_complex128 &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_complex64 \p rhs.
     */
    Scalar &operator=(const cytnx_complex64 &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_double \p rhs.
     */
    Scalar &operator=(const cytnx_double &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_float \p rhs.
     */
    Scalar &operator=(const cytnx_float &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_uint64 \p rhs.
     */
    Scalar &operator=(const cytnx_uint64 &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_int64 \p rhs.
     */
    Scalar &operator=(const cytnx_int64 &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_uint32 \p rhs.
     */
    Scalar &operator=(const cytnx_uint32 &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_int32 \p rhs.
     */
    Scalar &operator=(const cytnx_int32 &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_uint16 \p rhs.
     */
    Scalar &operator=(const cytnx_uint16 &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_int16 \p rhs.
     */
    Scalar &operator=(const cytnx_int16 &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_bool \p rhs.
     */
    Scalar &operator=(const cytnx_bool &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /**
     * @brief Type conversion function.
     * @param[in] dtype The type of the output Scalar (see cytnx::Type for more details).
     * @return The converted Scalar.
     * @attention The function cannot convert from complex to real, please use
     * cytnx::Scalar::real() or cytnx::Scalar::imag() to get the real or imaginary
     * part of the Scalar instead.
     */
    Scalar astype(const unsigned int &dtype) const {
      Scalar out(this->_impl->astype(dtype));
      return out;
    }

    /**
     * @brief Get the conjugate of the Scalar. That means return \f$ c^* \f$ if
     * the Scalar is \f$ c \f$.
     * @return The conjugate of the Scalar.
     */
    Scalar conj() const {
      Scalar out = *this;
      out._impl->conj_();
      return out;
    }

    /**
     * @brief Get the imaginary part of the Scalar. That means return \f$ \Im(c) \f$ if
     * the Scalar is \f$ c \f$.
     * @return The imaginary part of the Scalar.
     */
    Scalar imag() const { return Scalar(this->_impl->get_imag()); }

    /**
     * @brief Get the real part of the Scalar. That means return \f$ \Re(c) \f$ if
     * the Scalar is \f$ c \f$.
     * @return The real part of the Scalar.
     */
    Scalar real() const { return Scalar(this->_impl->get_real()); }
    // Scalar& set_imag(const Scalar &in){   return *this;}
    // Scalar& set_real(const Scalar &in){   return *this;}

    /**
     * @brief Get the dtype of the Scalar (see cytnx::Type for more details).
     */
    int dtype() const { return this->_impl->_dtype; }

    // print()
    /**
     * @brief Print the Scalar to the standard output.
     */
    void print() const {
      this->_impl->print(std::cout);
      std::cout << std::string(" Scalar dtype: [") << Type.getname(this->_impl->_dtype)
                << std::string("]") << std::endl;
    }

    // casting
    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_double.
    explicit operator cytnx_double() const { return this->_impl->to_cytnx_double(); }

    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_float
    explicit operator cytnx_float() const { return this->_impl->to_cytnx_float(); }

    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_uint64.
    explicit operator cytnx_uint64() const { return this->_impl->to_cytnx_uint64(); }

    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_int64.
    explicit operator cytnx_int64() const { return this->_impl->to_cytnx_int64(); }

    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_uint32.
    explicit operator cytnx_uint32() const { return this->_impl->to_cytnx_uint32(); }

    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_int32.
    explicit operator cytnx_int32() const { return this->_impl->to_cytnx_int32(); }

    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_uint16.
    explicit operator cytnx_uint16() const { return this->_impl->to_cytnx_uint16(); }

    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_int16.
    explicit operator cytnx_int16() const { return this->_impl->to_cytnx_int16(); }

    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_bool.
    explicit operator cytnx_bool() const { return this->_impl->to_cytnx_bool(); }

    /// @cond
    // destructor
    ~Scalar() {
      if (this->_impl != nullptr) delete this->_impl;
    };
    /// @endcond

    // arithmetic:
    ///@brief The addition assignment operator of the Scalar class with a given number (template).
    template <class T>
    void operator+=(const T &rc) {
      this->_impl->iadd(rc);
    }

    ///@brief The addition assignment operator of the Scalar class with a given Scalar.
    void operator+=(const Scalar &rhs) { this->_impl->iadd(rhs._impl); }

    ///@brief The subtraction assignment operator of the Scalar class with a given number
    ///(template).
    template <class T>
    void operator-=(const T &rc) {
      this->_impl->isub(rc);
    }

    ///@brief The subtraction assignment operator of the Scalar class with a given Scalar.
    void operator-=(const Scalar &rhs) { this->_impl->isub(rhs._impl); }
    template <class T>

    ///@brief The multiplication assignment operator of the Scalar class with a given number
    ///(template).
    void operator*=(const T &rc) {
      this->_impl->imul(rc);
    }

    ///@brief The multiplication assignment operator of the Scalar class with a given Scalar.
    void operator*=(const Scalar &rhs) { this->_impl->imul(rhs._impl); }
    template <class T>

    /**
     * @brief The division assignment operator of the Scalar class with a given number (template).
     */
    void operator/=(const T &rc) {
      this->_impl->idiv(rc);
    }

    /**
     * @brief The division assignment operator of the Scalar class with a given Scalar.
     */
    void operator/=(const Scalar &rhs) { this->_impl->idiv(rhs._impl); }

    /// @brief Set the Scalar to absolute value. (inplace)
    void iabs() { this->_impl->iabs(); }

    /// @brief Set the Scalar to square root. (inplace)
    void isqrt() { this->_impl->isqrt(); }

    /**
     * @brief The member function to get the absolute value of the Scalar.
     * @note Compare to the iabs() function, this function will return a new Scalar object.
     * @return The absolute value of the Scalar.
     * @see iabs()
     */
    Scalar abs() const {
      Scalar out = *this;
      out._impl->iabs();
      return out.real();
    }

    /**
     * @brief The member function to get the square root of the Scalar.
     * @note Compare to the isqrt() function, this function will return a new Scalar object.
     * @return The square root of the Scalar.
     * @see isqrt()
     */
    Scalar sqrt() const {
      Scalar out = *this;
      out._impl->isqrt();
      return out;
    }

    // comparison <
    /**
     * @brief Return whether the current Scalar is less than a given template number \p rc.
     * @details That is, whether \f$ s < r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given number \p rc.
     * @see operator<(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    bool less(const T &rc) const {
      Scalar tmp;
      int rid = Type.cy_typeid(rc);
      if (rid < this->dtype()) {
        tmp = this->astype(rid);
        return tmp._impl->less(rc);
      } else {
        return this->_impl->less(rc);
      }
    }

    /**
     * @brief Return whether the current Scalar is less than a given Scalar \p rhs.
     * @details That is, whether \f$ s < r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given Scalar \p rhs.
     * @see operator<(const Scalar &lhs, const Scalar &rhs)
     */
    bool less(const Scalar &rhs) const {
      Scalar tmp;
      if (rhs.dtype() < this->dtype()) {
        tmp = this->astype(rhs.dtype());
        return tmp._impl->less(rhs._impl);
      } else {
        return this->_impl->less(rhs._impl);
      }
    }

    // comparison <=

    /**
     * @brief Return whether the current Scalar is less than or equal to a given template number \p
     * rc.
     * @details That is, whether \f$ s \leq r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given number \p rc.
     * @see operator<=(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    bool leq(const T &rc) const {
      Scalar tmp;
      int rid = Type.cy_typeid(rc);
      if (rid < this->dtype()) {
        tmp = this->astype(rid);
        return !(tmp._impl->greater(rc));
      } else {
        return !(this->_impl->greater(rc));
      }
    }

    /**
     * @brief Return whether the current Scalar is less than or equal to a given Scalar \p rhs.
     * @details That is, whether \f$ s \leq r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given Scalar \p rhs.
     * @see operator<=(const Scalar &lhs, const Scalar &rhs)
     */
    bool leq(const Scalar &rhs) const {
      Scalar tmp;
      if (rhs.dtype() < this->dtype()) {
        tmp = this->astype(rhs.dtype());
        return !(tmp._impl->greater(rhs._impl));
      } else {
        return !(this->_impl->greater(rhs._impl));
      }
    }

    // comparison >
    /**
     * @brief Return whether the current Scalar is greater than a given template number \p rc.
     * @details That is, whether \f$ s > r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given number \p rc.
     * @see operator>(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    bool greater(const T &rc) const {
      Scalar tmp;
      int rid = Type.cy_typeid(rc);
      if (rid < this->dtype()) {
        tmp = this->astype(rid);
        return tmp._impl->greater(rc);
      } else {
        return this->_impl->greater(rc);
      }
    }

    /**
     * @brief Return whether the current Scalar is greater than a given Scalar \p rhs.
     * @details That is, whether \f$ s > r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given Scalar \p rhs.
     * @see operator>(const Scalar &lhs, const Scalar &rhs)
     */
    bool greater(const Scalar &rhs) const {
      Scalar tmp;
      if (rhs.dtype() < this->dtype()) {
        tmp = this->astype(rhs.dtype());
        return tmp._impl->greater(rhs._impl);
      } else {
        return this->_impl->greater(rhs._impl);
      }
    }

    // comparison >=

    /**
     * @brief Return whether the current Scalar is greater than or equal to a given template number
     * \p rc.
     * @details That is, whether \f$ s \geq r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given number \p rc.
     * @see operator>=(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    bool geq(const T &rc) const {
      Scalar tmp;
      int rid = Type.cy_typeid(rc);
      if (rid < this->dtype()) {
        tmp = this->astype(rid);
        return !(tmp._impl->less(rc));
      } else {
        return !(this->_impl->less(rc));
      }
    }

    /**
     * @brief Return whether the current Scalar is greater than or equal to a given Scalar \p rhs.
     * @details That is, whether \f$ s \geq r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given Scalar \p rhs.
     * @see operator>=(const Scalar &lhs, const Scalar &rhs)
     */
    bool geq(const Scalar &rhs) const {
      Scalar tmp;
      if (rhs.dtype() < this->dtype()) {
        tmp = this->astype(rhs.dtype());
        return !(tmp._impl->less(rhs._impl));
      } else {
        return !(this->_impl->less(rhs._impl));
      }
    }

    // comparison ==

    /**
     * @brief Return whether the current Scalar is equal to a given template number \p rc.
     * @details That is, whether \f$ s = r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given number \p rc.
     * @see operator==(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    bool eq(const T &rc) const {
      Scalar tmp;
      int rid = Type.cy_typeid(rc);
      if (rid < this->dtype()) {
        tmp = this->astype(rid);
        return tmp._impl->eq(rc);
      } else {
        return this->_impl->eq(rc);
      }
    }
    // /**
    //  * @brief Return whether the current Scalar is approximately equal to a given template number
    //  \p
    //  * rc.
    //  * @details That is, whether \f$ abs(s-r)<tol \f$, where \f$ s \f$ is the current Scalar
    //  * itself, \f$ r \f$ is the given number \p rc and \p tol is the tolerance value.
    //  */
    // template <class T>
    // bool approx_eq(const T &rc, cytnx_double tol = 1e-8) const {
    //   Scalar tmp;
    //   int rid = Type.cy_typeid(rc);
    //   if (rid < this->dtype()) {
    //     tmp = this->astype(rid);
    //     return tmp._impl->approx_eq(rc, tol);
    //   } else {
    //     return this->_impl->approx_eq(rc, tol);
    //   }
    // }

    /**
     * @brief Return whether the current Scalar is equal to a given Scalar \p rhs.
     * @details That is, whether \f$ s = r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given Scalar \p rhs.
     * @see operator==(const Scalar &lhs, const Scalar &rhs)
     */
    bool eq(const Scalar &rhs) const {
      Scalar tmp;
      if (rhs.dtype() < this->dtype()) {
        tmp = this->astype(rhs.dtype());
        return tmp._impl->eq(rhs._impl);
      } else {
        return this->_impl->eq(rhs._impl);
      }
    }
    // /**
    //  * @brief Return whether the current Scalar is approximately equal to a given Scalar \p rhs.
    //  * @details That is, whether \f$ abs(s-r)<tol \f$, where \f$ s \f$ is the current Scalar
    //  * itself, \f$ r \f$ is the given Scalar \p rhs and \p tol is the tolerance value.
    //  */
    // bool approx_eq(const Scalar &rhs, cytnx_double tol = 1e-8) const {
    //   Scalar tmp;
    //   if (rhs.dtype() < this->dtype()) {
    //     tmp = this->astype(rhs.dtype());
    //     return tmp._impl->approx_eq(rhs._impl, tol);
    //   } else {
    //     return this->_impl->approx_eq(rhs._impl, tol);
    //   }
    // }

    // radd: Scalar + c

    /**
     * @brief Return the addition of the current Scalar and a given template number \p rc.
     * @see operator+(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    Scalar radd(const T &rc) const {
      Scalar out;
      int rid = Type.cy_typeid(rc);
      if (this->dtype() < rid) {
        out = *this;
      } else {
        out = this->astype(rid);
      }
      out._impl->iadd(rc);
      return out;
    }

    /**
     * @brief Return the addition of the current Scalar and a given Scalar \p rhs.
     * @see operator+(const Scalar &lhs, const Scalar &rhs)
     */
    Scalar radd(const Scalar &rhs) const {
      Scalar out;
      if (this->dtype() < rhs.dtype()) {
        out = *this;
      } else {
        out = this->astype(rhs.dtype());
      }
      out._impl->iadd(rhs._impl);
      return out;
    }

    // rmul: Scalar * c

    /**
     * @brief Return the multiplication of the current Scalar and a given template number \p rc.
     * @see operator*(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    Scalar rmul(const T &rc) const {
      Scalar out;
      int rid = Type.cy_typeid(rc);
      if (this->dtype() < rid) {
        out = *this;
      } else {
        out = this->astype(rid);
      }
      out._impl->imul(rc);
      return out;
    }

    /**
     * @brief Return the multiplication of the current Scalar and a given Scalar \p rhs.
     * @see operator*(const Scalar &lhs, const Scalar &rhs)
     */
    Scalar rmul(const Scalar &rhs) const {
      Scalar out;
      if (this->dtype() < rhs.dtype()) {
        out = *this;
      } else {
        out = this->astype(rhs.dtype());
      }
      out._impl->imul(rhs._impl);
      return out;
    }

    // rsub: Scalar - c

    /**
     * @brief Return the subtraction of the current Scalar and a given template number \p rc.
     * @see operator-(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    Scalar rsub(const T &rc) const {
      Scalar out;
      int rid = Type.cy_typeid(rc);
      if (this->dtype() < rid) {
        out = *this;
      } else {
        out = this->astype(rid);
      }
      out._impl->isub(rc);
      return out;
    }

    /**
     * @brief Return the subtraction of the current Scalar and a given Scalar \p rhs.
     * @see operator-(const Scalar &lhs, const Scalar &rhs)
     */
    Scalar rsub(const Scalar &rhs) const {
      Scalar out;
      if (this->dtype() < rhs.dtype()) {
        out = *this;
      } else {
        out = this->astype(rhs.dtype());
      }
      out._impl->isub(rhs._impl);
      return out;
    }

    // rdiv: Scalar / c

    /**
     * @brief Return the division of the current Scalar and a given template number \p rc.
     * @see operator/(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    Scalar rdiv(const T &rc) const {
      Scalar out;
      int rid = Type.cy_typeid(rc);
      if (this->dtype() < rid) {
        out = *this;
      } else {
        out = this->astype(rid);
      }
      out._impl->idiv(rc);
      return out;
    }

    /**
     * @brief Return the division of the current Scalar and a given Scalar \p rhs.
     * @see operator/(const Scalar &lhs, const Scalar &rhs)
     */
    Scalar rdiv(const Scalar &rhs) const {
      Scalar out;
      if (this->dtype() < rhs.dtype()) {
        out = *this;
      } else {
        out = this->astype(rhs.dtype());
      }
      out._impl->idiv(rhs._impl);
      return out;
    }

    /*
    //operator:
    template<class T>
    Scalar operator+(const T &rc){
        return this->radd(rc);
    }
    template<class T>
    Scalar operator*(const T &rc){
        return this->rmul(rc);
    }
    template<class T>
    Scalar operator-(const T &rc){
        return this->rsub(rc);
    }
    template<class T>
    Scalar operator/(const T &rc){
        return this->rdiv(rc);
    }

    template<class T>
    bool operator<(const T &rc){
        return this->less(rc);
    }

    template<class T>
    bool operator>(const T &rc){
        return this->greater(rc);
    }


    template<class T>
    bool operator<=(const T &rc){
        return this->leq(rc);
    }

    template<class T>
    bool operator>=(const T &rc){
        return this->geq(rc);
    }
    */
  };
};
}  // namespace cytnx
#endif
#endif
