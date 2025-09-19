#ifndef CYTNX_INTRUSIVE_PTR_BASE_H_
#define CYTNX_INTRUSIVE_PTR_BASE_H_

#include <ostream>
#include <cassert>
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/checked_delete.hpp>
#include <boost/detail/atomic_count.hpp>

namespace cytnx {
  /// @cond
  class Storage_base;  // forward, defined in backend/Storage.hpp
  // The following two declarations are necessary for ADL.
  void intrusive_ptr_add_ref(Storage_base *);
  void intrusive_ptr_release(Storage_base *);

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

    // hook boost::intrusive_ptr add
    friend void intrusive_ptr_add_ref(T *s) {
      // add ref
      assert(s != nullptr);
      auto &base = static_cast<const intrusive_ptr_base<T> &>(*s);
      assert(base.ref_count >= 0);
      ++base.ref_count;
    }
    // hook boost::intrusive_ptr release
    friend void intrusive_ptr_release(T *s) {
      // release ref
      assert(s != nullptr);
      auto &base = static_cast<const intrusive_ptr_base<T> &>(*s);
      assert(base.ref_count > 0);
      if (--base.ref_count == 0) boost::checked_delete(static_cast<T const *>(s));
    }

    boost::intrusive_ptr<T> self() { return boost::intrusive_ptr<T>((T *)this); }

    boost::intrusive_ptr<const T> self() const {
      return boost::intrusive_ptr<const T>((T const *)this);
    }

    int refcount() const { return ref_count; }

   private:
    // should be modifiable within the class.
    mutable boost::detail::atomic_count ref_count;
  };
  ///@endcond
}  // namespace cytnx

#endif  // CYTNX_INTRUSIVE_PTR_BASE_H_
