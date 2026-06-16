#ifndef CYTNX_MDSPAN_HPP_
#define CYTNX_MDSPAN_HPP_

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace cytnx {
  namespace stdex {

    inline constexpr std::size_t dynamic_extent = static_cast<std::size_t>(-1);

    namespace detail {

      template <std::size_t... Values>
      struct count_dynamic_extents;

      template <>
      struct count_dynamic_extents<> {
        static constexpr std::size_t value = 0;
      };

      template <std::size_t First, std::size_t... Rest>
      struct count_dynamic_extents<First, Rest...> {
        static constexpr std::size_t value =
          (First == dynamic_extent ? 1 : 0) + count_dynamic_extents<Rest...>::value;
      };

      template <std::size_t Index, std::size_t First, std::size_t... Rest>
      struct static_extent_at {
        static constexpr std::size_t value = static_extent_at<Index - 1, Rest...>::value;
      };

      template <std::size_t First, std::size_t... Rest>
      struct static_extent_at<0, First, Rest...> {
        static constexpr std::size_t value = First;
      };

    }  // namespace detail

    template <class IndexType, std::size_t... Extents>
    class extents {
     public:
      using index_type = IndexType;
      using size_type = std::make_unsigned_t<index_type>;
      using rank_type = std::size_t;

      static_assert(std::is_integral_v<index_type>, "extents index_type must be integral");

      static constexpr rank_type rank() noexcept { return sizeof...(Extents); }
      static constexpr rank_type rank_dynamic() noexcept {
        return detail::count_dynamic_extents<Extents...>::value;
      }
      using dynamic_extents_type = std::array<index_type, rank_dynamic()>;

      constexpr extents() noexcept {
        static_assert(rank_dynamic() == 0, "default construction requires no dynamic extents");
      }

      template <class... IndexTypes,
                std::enable_if_t<sizeof...(IndexTypes) == rank_dynamic(), int> = 0>
      explicit constexpr extents(IndexTypes... dynamic_extents) noexcept
          : dynamic_extents_{static_cast<index_type>(dynamic_extents)...} {}

      explicit constexpr extents(const dynamic_extents_type& dynamic_extents) noexcept
          : dynamic_extents_(dynamic_extents) {}

      static constexpr std::size_t static_extent(rank_type r) noexcept {
        constexpr std::array<std::size_t, rank()> values{Extents...};
        return values[r];
      }

      constexpr index_type extent(rank_type r) const noexcept {
        constexpr std::array<std::size_t, rank()> values{Extents...};
        rank_type dynamic_index = 0;
        for (rank_type i = 0; i < r; ++i) {
          if (values[i] == dynamic_extent) ++dynamic_index;
        }
        return values[r] == dynamic_extent ? dynamic_extents_[dynamic_index]
                                           : static_cast<index_type>(values[r]);
      }

     private:
      dynamic_extents_type dynamic_extents_{};
    };

    namespace detail {

      template <class IndexType, std::size_t Rank, class Sequence>
      struct make_dextents;

      template <class IndexType, std::size_t Rank, std::size_t... Indices>
      struct make_dextents<IndexType, Rank, std::index_sequence<Indices...>> {
        using type = extents<IndexType, ((void)Indices, dynamic_extent)...>;
      };

    }  // namespace detail

    template <class IndexType, std::size_t Rank>
    using dextents =
      typename detail::make_dextents<IndexType, Rank, std::make_index_sequence<Rank>>::type;

    template <class Extents>
    class layout_right_mapping {
     public:
      using extents_type = Extents;
      using index_type = typename extents_type::index_type;
      using rank_type = typename extents_type::rank_type;

      constexpr layout_right_mapping() noexcept = default;
      explicit constexpr layout_right_mapping(const extents_type& extents) noexcept
          : extents_(extents) {}

      constexpr const extents_type& extents() const noexcept { return extents_; }
      static constexpr rank_type rank() noexcept { return extents_type::rank(); }

      constexpr index_type required_span_size() const noexcept {
        index_type size = 1;
        for (rank_type i = 0; i < rank(); ++i) size *= extents_.extent(i);
        return size;
      }

      constexpr index_type stride(rank_type r) const noexcept {
        index_type value = 1;
        for (rank_type i = rank(); i-- > r + 1;) value *= extents_.extent(i);
        return value;
      }

      template <class... Indices>
      constexpr index_type operator()(Indices... indices) const noexcept {
        static_assert(sizeof...(Indices) == rank(), "incorrect number of mdspan indices");
        std::array<index_type, rank()> idx{static_cast<index_type>(indices)...};
        index_type offset = 0;
        for (rank_type i = 0; i < rank(); ++i) offset += idx[i] * stride(i);
        return offset;
      }

     private:
      extents_type extents_{};
    };

    struct layout_right {
      template <class Extents>
      using mapping = layout_right_mapping<Extents>;
    };

    template <class Extents>
    class layout_stride_mapping {
     public:
      using extents_type = Extents;
      using index_type = typename extents_type::index_type;
      using rank_type = typename extents_type::rank_type;

      constexpr layout_stride_mapping() noexcept = default;
      constexpr layout_stride_mapping(const extents_type& extents,
                                      const std::array<index_type, extents_type::rank()>& strides)
          : extents_(extents), strides_(strides) {}

      constexpr const extents_type& extents() const noexcept { return extents_; }
      static constexpr rank_type rank() noexcept { return extents_type::rank(); }
      constexpr index_type stride(rank_type r) const noexcept { return strides_[r]; }

      constexpr index_type required_span_size() const noexcept {
        if constexpr (rank() == 0) {
          return 1;
        } else {
          index_type max_offset = 0;
          for (rank_type i = 0; i < rank(); ++i) {
            if (extents_.extent(i) == 0) return 0;
            max_offset += (extents_.extent(i) - 1) * strides_[i];
          }
          return max_offset + 1;
        }
      }

      template <class... Indices>
      constexpr index_type operator()(Indices... indices) const noexcept {
        static_assert(sizeof...(Indices) == rank(), "incorrect number of mdspan indices");
        std::array<index_type, rank()> idx{static_cast<index_type>(indices)...};
        index_type offset = 0;
        for (rank_type i = 0; i < rank(); ++i) offset += idx[i] * strides_[i];
        return offset;
      }

     private:
      extents_type extents_{};
      std::array<index_type, extents_type::rank()> strides_{};
    };

    struct layout_stride {
      template <class Extents>
      using mapping = layout_stride_mapping<Extents>;
    };

    template <class ElementType, class Extents, class LayoutPolicy = layout_right>
    class mdspan {
     public:
      using element_type = ElementType;
      using extents_type = Extents;
      using layout_type = LayoutPolicy;
      using mapping_type = typename layout_type::template mapping<extents_type>;
      using index_type = typename extents_type::index_type;
      using rank_type = typename extents_type::rank_type;
      using data_handle_type = element_type*;
      using reference = element_type&;

      constexpr mdspan() noexcept = default;

      constexpr mdspan(data_handle_type ptr, const mapping_type& mapping) noexcept
          : ptr_(ptr), mapping_(mapping) {}

      explicit constexpr mdspan(data_handle_type ptr, const extents_type& extents) noexcept
          : ptr_(ptr), mapping_(extents) {}

      template <class... IndexTypes,
                std::enable_if_t<sizeof...(IndexTypes) == extents_type::rank_dynamic(), int> = 0>
      explicit constexpr mdspan(data_handle_type ptr, IndexTypes... dynamic_extents) noexcept
          : ptr_(ptr), mapping_(extents_type(static_cast<index_type>(dynamic_extents)...)) {}

      static constexpr rank_type rank() noexcept { return extents_type::rank(); }
      static constexpr rank_type rank_dynamic() noexcept { return extents_type::rank_dynamic(); }

      constexpr index_type extent(rank_type r) const noexcept {
        return mapping_.extents().extent(r);
      }
      constexpr index_type stride(rank_type r) const noexcept { return mapping_.stride(r); }
      constexpr data_handle_type data_handle() const noexcept { return ptr_; }
      constexpr const mapping_type& mapping() const noexcept { return mapping_; }
      constexpr index_type required_span_size() const noexcept {
        return mapping_.required_span_size();
      }

      template <class... Indices>
      constexpr reference operator()(Indices... indices) const noexcept {
        static_assert(sizeof...(Indices) == rank(), "incorrect number of mdspan indices");
        return ptr_[mapping_(static_cast<index_type>(indices)...)];
      }

     private:
      data_handle_type ptr_ = nullptr;
      mapping_type mapping_{};
    };

  }  // namespace stdex
}  // namespace cytnx

#endif  // CYTNX_MDSPAN_HPP_
