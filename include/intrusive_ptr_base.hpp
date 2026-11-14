#ifndef CYTNX_INTRUSIVE_PTR_BASE_H_
#define CYTNX_INTRUSIVE_PTR_BASE_H_

#include <ostream>
#include <cassert>
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/checked_delete.hpp>
#include <boost/detail/atomic_count.hpp>

namespace cytnx {
  /// @cond
  template <class T>
  class intrusive_ptr_base;

  // The boost::intrusive_ptr hooks, found by argument-dependent lookup. Deduction through the base
  // class gives the derived type T, so intrusive_ptr_release() deletes the complete object.
  //
  // They are namespace-scope function templates on purpose. Defined as friends inside the class
  // template, they break nvcc's cudafe++ as soon as a constexpr destructor calls them, which
  // boost::intrusive_ptr's destructor is since Boost 1.89 (#1124).
  template <class T>
  void intrusive_ptr_add_ref(const intrusive_ptr_base<T> *s);
  template <class T>
  void intrusive_ptr_release(const intrusive_ptr_base<T> *s);

  template <class T>
  class intrusive_ptr_base {
   public:
    /// constructor
    intrusive_ptr_base()
        : ref_count(0){
            // pass
          };

    /// copy constructor
    intrusive_ptr_base(intrusive_ptr_base<T> const &) : ref_count(0) {
      // pass
    }

    // copy assignment
    intrusive_ptr_base &operator=(intrusive_ptr_base const &rhs) {
      // not allowed.
      return *this;
    }

    boost::intrusive_ptr<T> self() { return boost::intrusive_ptr<T>((T *)this); }

    boost::intrusive_ptr<const T> self() const {
      return boost::intrusive_ptr<const T>((T const *)this);
    }

    int refcount() const { return ref_count; }

   private:
    template <class U>
    friend void intrusive_ptr_add_ref(const intrusive_ptr_base<U> *s);
    template <class U>
    friend void intrusive_ptr_release(const intrusive_ptr_base<U> *s);

    // should be modifiable within the class.
    mutable boost::detail::atomic_count ref_count;
  };

  template <class T>
  void intrusive_ptr_add_ref(const intrusive_ptr_base<T> *s) {
    assert(s != nullptr);
    assert(s->ref_count >= 0);
    ++s->ref_count;
  }

  template <class T>
  void intrusive_ptr_release(const intrusive_ptr_base<T> *s) {
    assert(s != nullptr);
    assert(s->ref_count > 0);
    if (--s->ref_count == 0) boost::checked_delete(static_cast<T const *>(s));
  }
  ///@endcond
}  // namespace cytnx

#endif  // CYTNX_INTRUSIVE_PTR_BASE_H_
