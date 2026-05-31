#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_STRIDE_VIEW_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_STRIDE_VIEW_H_

#include <cstddef>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace cytnx {
  namespace linalg_internal {

    // A view over every step-th element of a random-access, sized range, e.g. the
    // diagonal of a matrix laid out with a fixed stride. It lets a single
    // reduction routine (PairwiseSum) consume both contiguous and strided
    // sequences.
    //
    // The iterator stores the underlying range's iterator rather than a pointer
    // to the view, so iterators stay valid for as long as the underlying data
    // lives, independent of the stride_view object's lifetime.
    template <std::ranges::view V>
    requires std::ranges::random_access_range<V> && std::ranges::sized_range<V>
    class stride_view : public std::ranges::view_interface<stride_view<V>> {
      V base_{};
      std::size_t step_ = 1;

     public:
      class iterator {
        using BaseIter = std::ranges::iterator_t<const V>;
        BaseIter base_{};
        std::size_t step_ = 1;
        std::size_t index_ = 0;

       public:
        using value_type = std::ranges::range_value_t<V>;
        using difference_type = std::ptrdiff_t;
        using iterator_concept = std::random_access_iterator_tag;
        using iterator_category = std::random_access_iterator_tag;
        using reference = std::ranges::range_reference_t<const V>;

        iterator() = default;
        iterator(BaseIter base, std::size_t step, std::size_t index)
            : base_(base), step_(step), index_(index) {}

        reference operator*() const { return base_[static_cast<difference_type>(index_ * step_)]; }
        reference operator[](difference_type n) const {
          return base_[static_cast<difference_type>((index_ + static_cast<std::size_t>(n)) *
                                                    step_)];
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
        iterator& operator-=(difference_type n) { return *this += -n; }

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
      };

      stride_view() = default;
      stride_view(V base, std::size_t step) : base_(std::move(base)), step_(step) {
        if (step_ == 0) throw std::invalid_argument("stride_view: stride must be positive");
      }

      std::size_t size() const {
        const auto n = static_cast<std::size_t>(std::ranges::size(base_));
        return n / step_ + (n % step_ != 0);
      }
      iterator begin() const { return iterator(std::ranges::begin(base_), step_, 0); }
      iterator end() const { return iterator(std::ranges::begin(base_), step_, size()); }
    };

    template <class R>
    stride_view(R&&, std::size_t) -> stride_view<std::views::all_t<R>>;

    // Range-adaptor closure so a strided view can be written as
    // `range | stride(step)`, e.g. `std::span<const T>(data, extent) | stride(k)`.
    struct stride_closure {
      std::size_t step;
      template <std::ranges::viewable_range R>
      friend auto operator|(R&& range, stride_closure closure) {
        return stride_view(std::views::all(std::forward<R>(range)), closure.step);
      }
    };
    inline stride_closure stride(std::size_t step) { return stride_closure{step}; }

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_STRIDE_VIEW_H_
