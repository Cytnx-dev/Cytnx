#ifndef _H_intrusive_ptr_base_
#define _H_intrusive_ptr_base_

#include <ostream>
#include <cassert>
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/checked_delete.hpp>
#include <boost/detail/atomic_count.hpp>

namespace cytnx {
  template <class T>
  /// @cond
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
    friend void intrusive_ptr_add_ref(intrusive_ptr_base<T> const *s) {
      // add ref
      // std::cout << "add" << std::endl;
      assert(s->ref_count >= 0);
      assert(s != 0);
      ++s->ref_count;
    }

    // hook boost::intrusive_ptr release
    friend void intrusive_ptr_release(intrusive_ptr_base<T> const *s) {
      // release ref
      // std::cout << "release" << std::endl;
      assert(s->ref_count > 0);
      assert(s != 0);
      if (--s->ref_count == 0) boost::checked_delete(static_cast<T const *>(s));
    }

    boost::intrusive_ptr<T> self() { return boost::intrusive_ptr<T>((T *)this); }

    boost::intrusive_ptr<const T> self() const {
      return boost::intrusive_ptr<const T>((T const *)this);
    }

    int refcount() const { return ref_count; }

   private:
    // should be modifialbe within the class.
    mutable boost::detail::atomic_count ref_count;
  };
  ///@endcond
}  // namespace cytnx

#endif
