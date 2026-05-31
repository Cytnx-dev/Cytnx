#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_STRIDE_VIEW_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_STRIDE_VIEW_H_

#include <cstddef>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace cytnx {
  namespace linalg_internal {

    /// @brief A view over every @p step -th element of a random-access, sized range.
    ///
    /// Wraps any sized random-access range @p V (e.g. `std::span<const T>`) and
    /// presents a fresh random-access, sized range whose @c i -th element is the
    /// @p step*i -th element of the underlying range. This lets a single
    /// reduction routine (e.g. `PairwiseSum`) consume both contiguous and
    /// strided sequences uniformly -- the diagonal of a matrix laid out with a
    /// fixed stride being the motivating use case.
    ///
    /// The iterator stores the underlying range's iterator rather than a
    /// pointer back to the view, so iterators stay valid for as long as the
    /// underlying data lives, independent of the @c stride_view object's
    /// lifetime.
    ///
    /// @par Example
    /// @code
    /// std::vector<double> v(30);
    /// // every third element: 0, 3, 6, ..., 27
    /// auto s = PairwiseSum(std::span<const double>(v) | stride(3));
    /// @endcode
    ///
    /// @tparam V An underlying view modelling @c std::ranges::view,
    ///           @c std::ranges::random_access_range, and
    ///           @c std::ranges::sized_range.
    template <std::ranges::view V>
    requires std::ranges::random_access_range<V> && std::ranges::sized_range<V>
    class stride_view : public std::ranges::view_interface<stride_view<V>> {
     public:
      /// @brief Random-access iterator over a @c stride_view.
      class iterator {
       public:
        using value_type = std::ranges::range_value_t<V>;
        using difference_type = std::ptrdiff_t;
        using iterator_concept = std::random_access_iterator_tag;
        using iterator_category = std::random_access_iterator_tag;
        using reference = std::ranges::range_reference_t<const V>;

        iterator() = default;
        iterator(std::ranges::iterator_t<const V> base, std::size_t step, std::size_t index)
            : base_(base), step_(step), index_(index) {}

        reference operator*() const { return base_[static_cast<difference_type>(index_ * step_)]; }
        reference operator[](difference_type n) const {
          return base_[(static_cast<difference_type>(index_) + n) *
                       static_cast<difference_type>(step_)];
        }

        iterator& operator++() {
          ++index_;
          return *this;
        }
        iterator operator++(int) {
          iterator t = *this;
          ++index_;
          return t;
        }
        iterator& operator--() {
          --index_;
          return *this;
        }
        iterator operator--(int) {
          iterator t = *this;
          --index_;
          return t;
        }

        iterator& operator+=(difference_type n) {
          index_ = static_cast<std::size_t>(static_cast<difference_type>(index_) + n);
          return *this;
        }
        iterator& operator-=(difference_type n) {
          // Direct subtraction; computing `*this += -n` would invoke UB when
          // n == std::numeric_limits<difference_type>::min() (the negation
          // cannot be represented in difference_type).
          index_ = static_cast<std::size_t>(static_cast<difference_type>(index_) - n);
          return *this;
        }

        friend iterator operator+(iterator it, difference_type n) {
          it += n;
          return it;
        }
        friend iterator operator+(difference_type n, iterator it) {
          it += n;
          return it;
        }
        friend iterator operator-(iterator it, difference_type n) {
          it -= n;
          return it;
        }
        friend difference_type operator-(const iterator& a, const iterator& b) {
          return static_cast<difference_type>(a.index_) - static_cast<difference_type>(b.index_);
        }

        friend bool operator==(const iterator& a, const iterator& b) {
          return a.index_ == b.index_;
        }
        friend auto operator<=>(const iterator& a, const iterator& b) {
          return a.index_ <=> b.index_;
        }

       private:
        std::ranges::iterator_t<const V> base_{};
        std::size_t step_ = 1;
        std::size_t index_ = 0;
      };

      stride_view() = default;

      /// @brief Construct a view that selects every @p step -th element of @p base.
      /// @param base The underlying random-access, sized range.
      /// @param step Positive stride; @p step == 0 is rejected.
      /// @throws std::invalid_argument if @p step is zero.
      stride_view(V base, std::size_t step) : base_(std::move(base)), step_(step) {
        if (step_ == 0) throw std::invalid_argument("stride_view: stride must be positive");
      }

      /// @brief Number of elements produced by this view.
      ///
      /// Equal to @c ceil(size(base) / step). An empty base or any base whose
      /// length is shorter than @c step yields a well-formed view that selects
      /// at most one element.
      std::size_t size() const {
        const auto n = static_cast<std::size_t>(std::ranges::size(base_));
        return n / step_ + (n % step_ != 0);
      }
      iterator begin() const { return iterator(std::ranges::begin(base_), step_, 0); }
      iterator end() const { return iterator(std::ranges::begin(base_), step_, size()); }

     private:
      V base_{};
      std::size_t step_ = 1;
    };

    template <class R>
    stride_view(R&&, std::size_t) -> stride_view<std::views::all_t<R>>;

    /// @brief Range-adaptor closure for the pipe form `r | stride(k)`.
    /// @see stride
    struct stride_closure {
      std::size_t step;
      template <std::ranges::viewable_range R>
      friend auto operator|(R&& range, stride_closure closure) {
        return stride_view(std::views::all(std::forward<R>(range)), closure.step);
      }
    };

    /// @brief Build a closure that pipes a range through @c stride_view.
    /// @param step Positive stride forwarded to @c stride_view's constructor.
    /// @par Example
    /// @code
    /// std::span<const T>(data, extent) | stride(diag_stride)
    /// @endcode
    inline stride_closure stride(std::size_t step) { return stride_closure{step}; }

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_STRIDE_VIEW_H_
