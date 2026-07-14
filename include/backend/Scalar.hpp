#ifndef CYTNX_BACKEND_SCALAR_H_
#define CYTNX_BACKEND_SCALAR_H_

#ifndef BACKEND_TORCH
  #include "Type.hpp"
  #include "cytnx_error.hpp"
  #include "intrusive_ptr_base.hpp"
  #include <ostream>
  #include <type_traits>
  #include <variant>
namespace cytnx {

  ///@cond
  class Storage_base;

  // The following two declarations are necessary for ADL.
  void intrusive_ptr_add_ref(Storage_base *);
  void intrusive_ptr_release(Storage_base *);
  ///@endcond

  // CytnxType<T> -- the 11 concrete value types a Scalar can hold (Type_list
  // minus Void) -- is defined in Type.hpp (included above). #979 landed it
  // there, so the copy this PR originally carried here is dropped to avoid a
  // redefinition. It constrains the templated Scalar API surface so non-cytnx
  // types get one crisp "constraint not satisfied" instead of an 11-way
  // overload ambiguity.

  /**
   * @brief A class to represent a scalar.
   * @details This class is used to represent a scalar. You can construct a Scalar by
   * a given value and a dtype (see cytnx::Type for available dtype).
   *
   * @details Implementation note: the value is held inline in a std::variant
   * (see #847), rather than a heap-allocated PIMPL hierarchy. std::monostate
   * represents the Void/uninitialized state. All promotion routes through
   * Type.type_promote() via std::visit; copy/move are `=default` (value
   * semantics, no manual heap bookkeeping -- fixes #935's self-assignment UB
   * and double-alloc copy ctor).
   */
  class Scalar {
   public:
    ///@cond
    // The list of alternatives mirrors Type_list (see Type.hpp), with
    // std::monostate standing in for "void" so an uninitialized Scalar is a
    // real, distinguishable state rather than a null pointer.
    using ScalarVariant =
      std::variant<std::monostate, cytnx_complex128, cytnx_complex64, cytnx_double, cytnx_float,
                   cytnx_int64, cytnx_uint64, cytnx_int32, cytnx_uint32, cytnx_int16, cytnx_uint16,
                   cytnx_bool>;

    // ScalarVariant is index-aligned with Type_list: alternative i holds the
    // type whose dtype id is i (monostate <-> void at index Type.Void == 0).
    // This is load-bearing -- dtype() is implemented as _val.index() -- so
    // pin it at compile time. Adding a dtype to Type_list without extending
    // ScalarVariant (or vice versa) fails right here.
    static_assert(std::variant_size_v<ScalarVariant> == N_Type,
                  "ScalarVariant must have exactly one alternative per Type_list entry");
    static_assert(
      std::is_same_v<std::variant_alternative_t<Type_class::ComplexDouble, ScalarVariant>,
                     cytnx_complex128> &&
        std::is_same_v<std::variant_alternative_t<Type_class::ComplexFloat, ScalarVariant>,
                       cytnx_complex64> &&
        std::is_same_v<std::variant_alternative_t<Type_class::Double, ScalarVariant>,
                       cytnx_double> &&
        std::is_same_v<std::variant_alternative_t<Type_class::Float, ScalarVariant>, cytnx_float> &&
        std::is_same_v<std::variant_alternative_t<Type_class::Int64, ScalarVariant>, cytnx_int64> &&
        std::is_same_v<std::variant_alternative_t<Type_class::Uint64, ScalarVariant>,
                       cytnx_uint64> &&
        std::is_same_v<std::variant_alternative_t<Type_class::Int32, ScalarVariant>, cytnx_int32> &&
        std::is_same_v<std::variant_alternative_t<Type_class::Uint32, ScalarVariant>,
                       cytnx_uint32> &&
        std::is_same_v<std::variant_alternative_t<Type_class::Int16, ScalarVariant>, cytnx_int16> &&
        std::is_same_v<std::variant_alternative_t<Type_class::Uint16, ScalarVariant>,
                       cytnx_uint16> &&
        std::is_same_v<std::variant_alternative_t<Type_class::Bool, ScalarVariant>, cytnx_bool>,
      "each ScalarVariant alternative must sit at its Type_list dtype index");

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

      // One constrained template replaces the former 11 per-type overloads;
      // it delegates through Scalar, whose set_item path they all shared.
      // (Defined inline: the enclosing Scalar class is complete inside
      // member bodies of a nested class.)
      template <CytnxType T>
      Sproxy &operator=(const T &rc) {
        return *this = Scalar(rc);
      }

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

    /// @cond
    // The stored value. std::monostate == Void/uninitialized.
    ScalarVariant _val;
    /// @endcond

    /// @brief default constructor
    Scalar() : _val(std::monostate{}) {}

    // init!!
    /// @brief init a Scalar with a cytnx::cytnx_complex128
    Scalar(const cytnx_complex128 &in) : _val(in) {}

    /// @brief init a Scalar with a cytnx::cytnx_complex64
    Scalar(const cytnx_complex64 &in) : _val(in) {}

    /// @brief init a Scalar with a cytnx::cytnx_double
    Scalar(const cytnx_double &in) : _val(in) {}

    /// @brief init a Scalar with a cytnx::cytnx_float
    Scalar(const cytnx_float &in) : _val(in) {}

    /// @brief init a Scalar with a cytnx::cytnx_uint64
    Scalar(const cytnx_uint64 &in) : _val(in) {}

    /// @brief init a Scalar with a cytnx::cytnx_int64
    Scalar(const cytnx_int64 &in) : _val(in) {}

    /// @brief init a Scalar with a cytnx::cytnx_uint32
    Scalar(const cytnx_uint32 &in) : _val(in) {}

    /// @brief init a Scalar with a cytnx::cytnx_int32
    Scalar(const cytnx_int32 &in) : _val(in) {}

    /// @brief init a Scalar with a cytnx::cytnx_uint16
    Scalar(const cytnx_uint16 &in) : _val(in) {}

    /// @brief init a Scalar with a cytnx::cytnx_int16
    Scalar(const cytnx_int16 &in) : _val(in) {}

    /// @brief init a Scalar with a cytnx::cytnx_bool
    Scalar(const cytnx_bool &in) : _val(in) {}

    /**
     * @brief Get the max value of the Scalar with the given \p dtype.
     * @details This function is used to get the max value of the Scalar with the given \p dtype.
     * That is, for example, if you want to get the max value of a Scalar with
     * \p dtype = cytnx::Type.Int16, then you will get the max value of a 16-bit integer 32767.
     * @param[in] dtype The data type of the Scalar.
     * @return The max value of the Scalar with the given \p dtype.
     */
    static Scalar maxval(const unsigned int &dtype);

    /**
     * @brief Get the min value of the Scalar with the given \p dtype.
     * @details This function is used to get the min value of the Scalar with the given \p dtype.
     * That is, for example, if you want to get the min value of a Scalar with
     * \p dtype = cytnx::Type.Int16, then you will get the min value of a 16-bit integer -32768.
     * @param[in] dtype The data type of the Scalar.
     * @return The min value of the Scalar with the given \p dtype.
     * @note For floating-point dtypes this returns the most negative finite value
     * (`std::numeric_limits<T>::lowest()`), not the smallest positive normal
     * value that `::min()` would give (fixes #935).
     */
    static Scalar minval(const unsigned int &dtype);

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
    Scalar(const T &in, const unsigned int &dtype) {
      this->Init_by_number(in);
      *this = this->astype(dtype);
    }

    /// @cond
    // move sproxy when use to get elements here.
    Scalar(const Sproxy &prox);
    /// @endcond

    // specialization of init:
    ///@cond
    void Init_by_number(const cytnx_complex128 &in) { this->_val = in; }
    void Init_by_number(const cytnx_complex64 &in) { this->_val = in; }
    void Init_by_number(const cytnx_double &in) { this->_val = in; }
    void Init_by_number(const cytnx_float &in) { this->_val = in; }
    void Init_by_number(const cytnx_int64 &in) { this->_val = in; }
    void Init_by_number(const cytnx_uint64 &in) { this->_val = in; }
    void Init_by_number(const cytnx_int32 &in) { this->_val = in; }
    void Init_by_number(const cytnx_uint32 &in) { this->_val = in; }
    void Init_by_number(const cytnx_int16 &in) { this->_val = in; }
    void Init_by_number(const cytnx_uint16 &in) { this->_val = in; }
    void Init_by_number(const cytnx_bool &in) { this->_val = in; }
    ///@endcond

    // Copy/move/assign are trivial value-type operations now: default them.
    // This fixes #935's self-assignment UB (the old hand-rolled operator=
    // deleted _impl before reading rhs._impl, and did not check for
    // self-assignment) and the double-allocation copy ctor.
    Scalar(const Scalar &rhs) = default;
    Scalar(Scalar &&rhs) = default;
    Scalar &operator=(const Scalar &rhs) = default;
    Scalar &operator=(Scalar &&rhs) = default;

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
    Scalar astype(const unsigned int &dtype) const;

    /**
     * @brief Get the conjugate of the Scalar. That means return \f$ c^* \f$ if
     * the Scalar is \f$ c \f$.
     * @return The conjugate of the Scalar.
     */
    Scalar conj() const;

    /**
     * @brief Get the imaginary part of the Scalar. That means return \f$ \Im(c) \f$ if
     * the Scalar is \f$ c \f$.
     * @return The imaginary part of the Scalar.
     */
    Scalar imag() const;

    /**
     * @brief Get the real part of the Scalar. That means return \f$ \Re(c) \f$ if
     * the Scalar is \f$ c \f$.
     * @return The real part of the Scalar.
     */
    Scalar real() const;

    /**
     * @brief Get the dtype of the Scalar (see cytnx::Type for more details).
     */
    int dtype() const;

    // print()
    /**
     * @brief Print the Scalar to the standard output.
     */
    void print() const;

    // casting
    /**
     * @brief The explicit casting operator of the Scalar class to any cytnx
     * value type (see cytnx::Type).
     * @details One constrained template replaces the former 9 per-type cast
     * operators (and extends the castable set to the complex types, which
     * were previously reachable only through named accessors). The
     * `requires` clause admits exactly the conversions the language can
     * perform on the held value: complex -> real has no static_cast, so it
     * errors with a hint to use real()/imag(); a Void (monostate) Scalar
     * converts to nothing and errors naming the dtypes.
     */
    template <CytnxType To>
    explicit operator To() const {
      return std::visit(
        []<typename U>(const U &from) -> To {
          if constexpr (requires { static_cast<To>(from); }) {
            return static_cast<To>(from);
          } else if constexpr (is_complex_v<U>) {
            cytnx_error_msg(true,
                            "[ERROR] Cannot convert Scalar from dtype [%s] to dtype [%s]. Use "
                            "real() or imag() to extract a component first.%s",
                            Type_enum_name<U>, Type_enum_name<To>, "\n");
            return To{};
          } else {
            cytnx_error_msg(true, "[ERROR] Cannot convert Scalar from dtype [%s] to dtype [%s].%s",
                            Type_enum_name<U>, Type_enum_name<To>, "\n");
            return To{};
          }
        },
        _val);
    }

    // Named complex accessors, kept for the existing internal/public callers;
    // they now simply delegate to the cast-operator template above.
    /// @cond
    cytnx_complex128 to_complex128() const { return static_cast<cytnx_complex128>(*this); }
    cytnx_complex64 to_complex64() const { return static_cast<cytnx_complex64>(*this); }
    ///@endcond

    /// @cond
    // destructor: trivial, variant handles cleanup.
    ~Scalar() = default;
    /// @endcond

    // arithmetic:
    ///@brief The addition assignment operator of the Scalar class with a given number (template).
    template <class T>
    void operator+=(const T &rc) {
      *this += Scalar(rc);
    }

    ///@brief The addition assignment operator of the Scalar class with a given Scalar.
    void operator+=(const Scalar &rhs);

    ///@brief The subtraction assignment operator of the Scalar class with a given number
    ///(template).
    template <class T>
    void operator-=(const T &rc) {
      *this -= Scalar(rc);
    }

    ///@brief The subtraction assignment operator of the Scalar class with a given Scalar.
    void operator-=(const Scalar &rhs);

    ///@brief The multiplication assignment operator of the Scalar class with a given number
    ///(template).
    template <class T>
    void operator*=(const T &rc) {
      *this *= Scalar(rc);
    }

    ///@brief The multiplication assignment operator of the Scalar class with a given Scalar.
    void operator*=(const Scalar &rhs);

    /**
     * @brief The division assignment operator of the Scalar class with a given number (template).
     */
    template <class T>
    void operator/=(const T &rc) {
      *this /= Scalar(rc);
    }

    /**
     * @brief The division assignment operator of the Scalar class with a given Scalar.
     */
    void operator/=(const Scalar &rhs);

    /// @brief Set the Scalar to absolute value. (inplace)
    void iabs();

    /// @brief Set the Scalar to square root. (inplace)
    void isqrt();

    /**
     * @brief The member function to get the absolute value of the Scalar.
     * @note Compare to the iabs() function, this function will return a new Scalar object.
     * @details The result always has a real (non-complex) dtype: abs() of a complex
     * Scalar returns the Float/Double magnitude, not a complex value with a zero
     * imaginary part (fixes the iabs()/abs() dtype inconsistency documented in #935).
     * @return The absolute value of the Scalar.
     * @see iabs()
     */
    Scalar abs() const;

    /**
     * @brief The member function to get the square root of the Scalar.
     * @note Compare to the isqrt() function, this function will return a new Scalar object.
     * @return The square root of the Scalar.
     * @see isqrt()
     */
    Scalar sqrt() const;

    // comparison <
    /**
     * @brief Return whether the current Scalar is less than a given template number \p rc.
     * @details That is, whether \f$ s < r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given number \p rc.
     * @see operator<(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    bool less(const T &rc) const {
      return this->less(Scalar(rc));
    }

    /**
     * @brief Return whether the current Scalar is less than a given Scalar \p rhs.
     * @details That is, whether \f$ s < r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given Scalar \p rhs.
     * @see operator<(const Scalar &lhs, const Scalar &rhs)
     */
    bool less(const Scalar &rhs) const;

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
      return this->leq(Scalar(rc));
    }

    /**
     * @brief Return whether the current Scalar is less than or equal to a given Scalar \p rhs.
     * @details That is, whether \f$ s \leq r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given Scalar \p rhs.
     * @see operator<=(const Scalar &lhs, const Scalar &rhs)
     */
    bool leq(const Scalar &rhs) const;

    // comparison >
    /**
     * @brief Return whether the current Scalar is greater than a given template number \p rc.
     * @details That is, whether \f$ s > r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given number \p rc.
     * @see operator>(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    bool greater(const T &rc) const {
      return this->greater(Scalar(rc));
    }

    /**
     * @brief Return whether the current Scalar is greater than a given Scalar \p rhs.
     * @details That is, whether \f$ s > r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given Scalar \p rhs.
     * @see operator>(const Scalar &lhs, const Scalar &rhs)
     */
    bool greater(const Scalar &rhs) const;

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
      return this->geq(Scalar(rc));
    }

    /**
     * @brief Return whether the current Scalar is greater than or equal to a given Scalar \p rhs.
     * @details That is, whether \f$ s \geq r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given Scalar \p rhs.
     * @see operator>=(const Scalar &lhs, const Scalar &rhs)
     */
    bool geq(const Scalar &rhs) const;

    // comparison ==

    /**
     * @brief Return whether the current Scalar is equal to a given template number \p rc.
     * @details That is, whether \f$ s = r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given number \p rc.
     * @see operator==(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    bool eq(const T &rc) const {
      return this->eq(Scalar(rc));
    }

    /**
     * @brief Return whether the current Scalar is equal to a given Scalar \p rhs.
     * @details That is, whether \f$ s = r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given Scalar \p rhs.
     * @see operator==(const Scalar &lhs, const Scalar &rhs)
     */
    bool eq(const Scalar &rhs) const;

    // radd: Scalar + c

    /**
     * @brief Return the addition of the current Scalar and a given template number \p rc.
     * @see operator+(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    Scalar radd(const T &rc) const {
      return this->radd(Scalar(rc));
    }

    /**
     * @brief Return the addition of the current Scalar and a given Scalar \p rhs.
     * @see operator+(const Scalar &lhs, const Scalar &rhs)
     */
    Scalar radd(const Scalar &rhs) const;

    // rmul: Scalar * c

    /**
     * @brief Return the multiplication of the current Scalar and a given template number \p rc.
     * @see operator*(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    Scalar rmul(const T &rc) const {
      return this->rmul(Scalar(rc));
    }

    /**
     * @brief Return the multiplication of the current Scalar and a given Scalar \p rhs.
     * @see operator*(const Scalar &lhs, const Scalar &rhs)
     */
    Scalar rmul(const Scalar &rhs) const;

    // rsub: Scalar - c

    /**
     * @brief Return the subtraction of the current Scalar and a given template number \p rc.
     * @see operator-(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    Scalar rsub(const T &rc) const {
      return this->rsub(Scalar(rc));
    }

    /**
     * @brief Return the subtraction of the current Scalar and a given Scalar \p rhs.
     * @see operator-(const Scalar &lhs, const Scalar &rhs)
     */
    Scalar rsub(const Scalar &rhs) const;

    // rdiv: Scalar / c

    /**
     * @brief Return the division of the current Scalar and a given template number \p rc.
     * @see operator/(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    Scalar rdiv(const T &rc) const {
      return this->rdiv(Scalar(rc));
    }

    /**
     * @brief Return the division of the current Scalar and a given Scalar \p rhs.
     * @see operator/(const Scalar &lhs, const Scalar &rhs)
     */
    Scalar rdiv(const Scalar &rhs) const;
  };

  // ladd: c + Scalar:

  /**
   * @brief The addition operator between two Scalar objects.
   * @details Return
   * \f[ l+r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  Scalar operator+(const Scalar &lc, const Scalar &rs);

  // lmul c * Scalar;
  /**
   * @brief The multiplication operator between two Scalar objects.
   * @details Return
   * \f[ l \cdot r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  Scalar operator*(const Scalar &lc, const Scalar &rs);

  // lsub c * Scalar;
  /**
   * @brief The subtraction operator between two Scalar objects.
   * @details Return
   * \f[ l-r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  Scalar operator-(const Scalar &lc, const Scalar &rs);

  // ldiv c / Scalar;
  /**
   * @brief The division operator between two Scalar objects.
   * @details Return
   * \f[ l/r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  Scalar operator/(const Scalar &lc, const Scalar &rs);

  // lless c < Scalar;
  /**
   * @brief The less-than operator between two Scalar objects.
   * @details Return
   * \f[ l<r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  bool operator<(const Scalar &lc, const Scalar &rs);

  // lgreater c > Scalar;
  /**
   * @brief The greater-than operator between two Scalar objects.
   * @details Return
   * \f[ l>r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  bool operator>(const Scalar &lc, const Scalar &rs);

  // lless c <= Scalar;
  /**
   * @brief The less-than-or-equal operator between two Scalar objects.
   * @details Return
   * \f[ l\leq r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  bool operator<=(const Scalar &lc, const Scalar &rs);

  // lgreater c >= Scalar;
  /**
   * @brief The greater-than-or-equal operator between two Scalar objects.
   * @details Return
   * \f[ l\geq r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  bool operator>=(const Scalar &lc, const Scalar &rs);

  // eq c == Scalar;
  /**
   * @brief The equal operator between two Scalar objects.
   * @details Return
   * \f[ l==r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  bool operator==(const Scalar &lc, const Scalar &rs);

  // abs:

  /**
   * @brief Return the absolute value of a Scalar object.
   * @details Return
   * \f[ \left|c\right| \f],
   * where \f$ c \f$ is the input Scalar \p c .
   */
  Scalar abs(const Scalar &c);

  // sqrt:
  /**
   * @brief Return the square root of a Scalar object.
   * @details Return
   * \f[ \sqrt{c} \f],
   * where \f$ c \f$ is the input Scalar \p c .
   */
  Scalar sqrt(const Scalar &c);

  // complex conversion:
  /// @brief Convert a Scalar object to a cytnx::complex128.
  cytnx_complex128 complex128(const Scalar &in);

  /// @brief Convert a Scalar object to a cytnx::complex64.
  cytnx_complex64 complex64(const Scalar &in);

  /// @cond
  std::ostream &operator<<(std::ostream &os, const Scalar &in);
  /// @endcond

}  // namespace cytnx

#endif

#endif  // CYTNX_BACKEND_SCALAR_H_
