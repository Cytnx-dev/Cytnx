#ifndef CYTNX_TENSORT_HPP_
#define CYTNX_TENSORT_HPP_

#include "Tensor.hpp"
#include "TensorT_cpu.hpp"
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "mdspan.hpp"

#ifdef UNI_GPU
  #include "TensorT_gpu.hpp"
#endif

#include "boost/smart_ptr/intrusive_ptr.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace cytnx {

  /**
   * @brief Shared typed owner for the data pointer used by TensorT.
   *
   * This is currently backed by a `std::shared_ptr<T>`. For views created from legacy `Tensor`, the
   * shared pointer aliases the Tensor storage and uses a deleter that keeps the legacy storage
   * reference count alive.
   */
  template <class T>
  class data_owner {
   public:
    data_owner() = default;
    explicit data_owner(std::shared_ptr<T> data) : data_(std::move(data)) {}

    T *get() const noexcept { return data_.get(); }
    T *operator->() const noexcept { return data_.get(); }
    explicit operator bool() const noexcept { return static_cast<bool>(data_); }

    const std::shared_ptr<T> &shared_ptr() const noexcept { return data_; }

   private:
    std::shared_ptr<T> data_;
  };

  /**
   * @brief Typed, ranked Tensor view with shared ownership of its data pointer.
   *
   * `TensorT` is intended as an internal kernel-facing representation. The element type, rank,
   * access policy, and layout are part of the C++ type, while `owner()` keeps the pointed-to data
   * alive independently of any legacy `Tensor` object used to create the view.
   */
  template <class T, std::size_t Rank, class Access, class Layout = stdex::layout_stride>
  class TensorT {
   public:
    using element_type = T;
    using access_type = Access;
    using layout_type = Layout;
    using extents_type = stdex::dextents<std::size_t, Rank>;
    using view_type = stdex::mdspan<T, extents_type, Layout>;
    using mapping_type = typename view_type::mapping_type;
    using owner_type = data_owner<T>;

    static constexpr std::size_t rank() noexcept { return Rank; }

    TensorT() = default;
    TensorT(owner_type owner, view_type view, Access access = Access{})
        : owner_(std::move(owner)), view_(view), access_(access) {}
    explicit TensorT(const T &value, Access access = Access{}) requires(Rank == 0)
        : TensorT(allocate_scalar(value), access) {
      static_assert(std::same_as<Access, host_access>,
                    "Direct TensorT allocation currently supports host_access only");
    }
    explicit TensorT(const std::array<std::size_t, Rank> &extents, Access access = Access{})
        : TensorT(allocate(extents), access) {
      static_assert(std::same_as<Access, host_access>,
                    "Direct TensorT allocation currently supports host_access only");
    }
    explicit TensorT(std::initializer_list<std::size_t> extents, Access access = Access{})
        : TensorT(extents_from_list(extents), access) {
      static_assert(std::same_as<Access, host_access>,
                    "Direct TensorT allocation currently supports host_access only");
    }

    std::size_t extent(std::size_t axis) const noexcept { return view_.extent(axis); }
    std::size_t stride(std::size_t axis) const noexcept { return view_.stride(axis); }
    std::size_t required_span_size() const noexcept { return view_.required_span_size(); }
    std::size_t size() const noexcept requires(Rank == 1) { return extent(0); }
    std::size_t rows() const noexcept requires(Rank == 2) { return extent(0); }
    std::size_t cols() const noexcept requires(Rank == 2) { return extent(1); }

    T *data() const noexcept { return view_.data_handle(); }
    T *data_handle() const noexcept { return view_.data_handle(); }

    const view_type &view() const noexcept { return view_; }
    const owner_type &owner() const noexcept { return owner_; }
    const Access &access() const noexcept { return access_; }

    static constexpr unsigned int dtype() { return Type_class::cy_typeid_v<std::remove_cv_t<T>>; }
    int device() const { return tensor_t_detail::access_device(access_); }
    T &value() const noexcept requires(Rank == 0) { return view_(); }

    template <class... Indices>
    T &operator()(Indices... indices) const noexcept {
      return view_(indices...);
    }

   private:
    struct allocated_view {
      owner_type owner;
      view_type view;
    };

    explicit TensorT(allocated_view allocated, Access access)
        : owner_(std::move(allocated.owner)), view_(allocated.view), access_(access) {}

    static std::array<std::size_t, Rank> extents_from_list(
      std::initializer_list<std::size_t> extents) {
      cytnx_error_msg(extents.size() != Rank,
                      "[ERROR] TensorT rank-%llu allocation received %llu extents.%s",
                      static_cast<unsigned long long>(Rank),
                      static_cast<unsigned long long>(extents.size()), "\n");
      std::array<std::size_t, Rank> out{};
      std::copy(extents.begin(), extents.end(), out.begin());
      return out;
    }

    static mapping_type make_contiguous_mapping(const std::array<std::size_t, Rank> &extents) {
      const extents_type md_extents(extents);
      if constexpr (std::same_as<Layout, stdex::layout_right>) {
        return mapping_type(md_extents);
      } else if constexpr (std::same_as<Layout, stdex::layout_stride>) {
        std::array<std::size_t, Rank> strides{};
        std::size_t step = 1;
        for (std::size_t i = Rank; i-- > 0;) {
          strides[i] = step;
          step *= extents[i];
        }
        return mapping_type(md_extents, strides);
      } else {
        static_assert(
          std::same_as<Layout, stdex::layout_right> || std::same_as<Layout, stdex::layout_stride>,
          "Unsupported TensorT layout for direct allocation");
      }
    }

    static allocated_view allocate(const std::array<std::size_t, Rank> &extents) {
      const mapping_type mapping = make_contiguous_mapping(extents);
      auto owner = owner_type(
        std::shared_ptr<T>(new T[mapping.required_span_size()](), std::default_delete<T[]>()));
      return allocated_view{owner, view_type(owner.get(), mapping)};
    }

    static allocated_view allocate_scalar(const T &value) {
      std::array<std::size_t, Rank> extents{};
      allocated_view allocated = allocate(extents);
      allocated.view() = value;
      return allocated;
    }

    owner_type owner_;
    view_type view_;
    [[no_unique_address]] Access access_{};
  };

  template <class T, std::size_t Rank, class Layout = stdex::layout_stride>
  using HostTensorT = TensorT<T, Rank, host_access, Layout>;

#ifdef UNI_GPU
  template <class T, std::size_t Rank, class Layout = stdex::layout_stride>
  using CudaTensorT = TensorT<T, Rank, cuda_access, Layout>;
#endif

  template <class T, class Access, class Layout = stdex::layout_stride>
  using VectorT = TensorT<T, 1, Access, Layout>;

  template <class T, class Access, class Layout = stdex::layout_stride>
  using MatrixT = TensorT<T, 2, Access, Layout>;

  template <class T, std::size_t Rank, class Layout = stdex::layout_stride>
  using TensorDeviceT = std::variant<HostTensorT<T, Rank, Layout>
#ifdef UNI_GPU
                                     ,
                                     CudaTensorT<T, Rank, Layout>
#endif
                                     >;

  namespace tensor_t_detail {

    /**
     * @brief Deleter used when a TensorT view aliases legacy Cytnx Storage.
     *
     * The deleter does not delete the raw pointer directly. It captures the legacy intrusive
     * storage handle so the allocation stays alive for the lifetime of the typed shared pointer.
     * `std::get_deleter` can recover this object for no-copy conversion back to legacy `Tensor`.
     */
    template <typename T>
    class legacy_storage_deleter {
     public:
      legacy_storage_deleter() = default;
      explicit legacy_storage_deleter(boost::intrusive_ptr<Storage_base> storage)
          : storage_(std::move(storage)) {}

      void operator()(T *) noexcept { storage_.reset(); }

      const boost::intrusive_ptr<Storage_base> &storage() const noexcept { return storage_; }

     private:
      boost::intrusive_ptr<Storage_base> storage_;
    };

    template <typename T, std::size_t Rank>
    void check_tensor_type_and_rank(const Tensor &tensor) {
      using element_type = std::remove_cv_t<T>;
      cytnx_error_msg(tensor.dtype() != Type_class::cy_typeid_v<element_type>,
                      "[ERROR] Attempt to convert dtype %d (%s) to TensorT of type %s",
                      tensor.dtype(), Type_class::getname(tensor.dtype()).c_str(),
                      Type_class::getname(Type_class::cy_typeid_v<element_type>).c_str());
      cytnx_error_msg(tensor.rank() != Rank,
                      "[ERROR] Attempt to view rank-%llu Tensor as rank-%llu TensorT.%s",
                      static_cast<unsigned long long>(tensor.rank()),
                      static_cast<unsigned long long>(Rank), "\n");
    }

    template <std::size_t Rank>
    std::array<std::size_t, Rank> extents_from_tensor(const Tensor &tensor) {
      std::array<std::size_t, Rank> extents{};
      for (std::size_t i = 0; i < Rank; ++i) {
        extents[i] = static_cast<std::size_t>(tensor._impl->shape()[i]);
      }
      return extents;
    }

    template <std::size_t Rank>
    std::array<std::size_t, Rank> strides_from_tensor(
      const Tensor &tensor, const std::array<std::size_t, Rank> &extents) {
      std::array<std::size_t, Rank> strides{};
      std::size_t step = 1;
      for (std::size_t i = Rank; i-- > 0;) {
        const std::size_t axis = static_cast<std::size_t>(tensor._impl->invmapper()[i]);
        cytnx_error_msg(axis >= Rank, "[ERROR] Invalid Tensor mapper metadata.%s", "\n");
        strides[axis] = step;
        step *= extents[axis];
      }
      return strides;
    }

    template <typename T>
    data_owner<T> owner_from_tensor(Tensor &tensor) {
      auto storage = tensor._impl->storage()._impl;
      T *data = tensor.ptr_as<T>();
      return data_owner<T>(std::shared_ptr<T>(data, legacy_storage_deleter<T>(std::move(storage))));
    }

    template <typename T>
    boost::intrusive_ptr<Storage_base> legacy_storage_from_owner(const data_owner<T> &owner) {
      auto *deleter = std::get_deleter<legacy_storage_deleter<T>>(owner.shared_ptr());
      cytnx_error_msg(deleter == nullptr,
                      "[ERROR] Cannot create a Tensor view from TensorT without legacy storage.%s",
                      "\n");
      return deleter->storage();
    }

    /**
     * @brief Return the logical axes ordered by physical contiguous memory order.
     *
     * This only accepts stride patterns that are exactly a contiguous row-major layout up to a
     * permutation of axes. General strided or offset views are not representable by current Cytnx
     * Tensor mapper metadata and are rejected.
     */
    template <class TensorView>
    std::vector<cytnx_uint64> memory_order_from_strides(const TensorView &view) {
      constexpr std::size_t rank = TensorView::rank();
      cytnx_error_msg(rank == 0, "[ERROR] Cannot convert rank-0 TensorT to Tensor.%s", "\n");

      std::array<bool, rank> used{};
      std::vector<cytnx_uint64> inner_to_outer;
      inner_to_outer.reserve(rank);

      std::size_t expected_stride = 1;
      for (std::size_t step = 0; step < rank; ++step) {
        bool found = false;
        for (std::size_t axis = 0; axis < rank; ++axis) {
          if (!used[axis] && view.stride(axis) == expected_stride) {
            used[axis] = true;
            inner_to_outer.push_back(static_cast<cytnx_uint64>(axis));
            expected_stride *= view.extent(axis);
            found = true;
            break;
          }
        }
        cytnx_error_msg(!found, "[ERROR] TensorT strides are not a contiguous permutation.%s",
                        "\n");
      }

      std::vector<cytnx_uint64> memory_order;
      memory_order.reserve(rank);
      for (std::size_t i = inner_to_outer.size(); i-- > 0;) {
        memory_order.push_back(inner_to_outer[i]);
      }
      return memory_order;
    }

  }  // namespace tensor_t_detail

  /**
   * @brief Create a storage-owning typed TensorT view preserving the Tensor's current layout.
   *
   * The returned TensorT keeps the Tensor storage alive independently of the legacy Tensor object.
   * The requested element type, rank, and backend access policy must match the Tensor metadata.
   */
  template <typename T, std::size_t Rank, class Access = host_access>
  TensorT<T, Rank, Access, stdex::layout_stride> make_tensor_t(Tensor &tensor) {
    using extents_type = stdex::dextents<std::size_t, Rank>;
    using mapping_type = typename stdex::layout_stride::template mapping<extents_type>;
    using view_type = stdex::mdspan<T, extents_type, stdex::layout_stride>;

    tensor_t_detail::check_tensor_type_and_rank<T, Rank>(tensor);
    auto access = tensor_t_detail::make_access<Access>(tensor.device());
    const auto extents = tensor_t_detail::extents_from_tensor<Rank>(tensor);
    const auto strides = tensor_t_detail::strides_from_tensor<Rank>(tensor, extents);
    auto owner = tensor_t_detail::owner_from_tensor<T>(tensor);
    view_type view(owner.get(), mapping_type(extents_type(extents), strides));
    return TensorT<T, Rank, Access, stdex::layout_stride>(std::move(owner), view, access);
  }

  /**
   * @brief Create a storage-owning typed layout-right TensorT view.
   *
   * This is a mutating function: it first calls `contiguous_()` on the legacy Tensor, so a
   * non-contiguous Tensor may receive new contiguous storage.
   */
  template <typename T, std::size_t Rank, class Access = host_access>
  TensorT<T, Rank, Access, stdex::layout_right> make_right_tensor_t(Tensor &tensor) {
    using extents_type = stdex::dextents<std::size_t, Rank>;
    using view_type = stdex::mdspan<T, extents_type, stdex::layout_right>;

    tensor_t_detail::check_tensor_type_and_rank<T, Rank>(tensor);
    auto access = tensor_t_detail::make_access<Access>(tensor.device());
    tensor.contiguous_();
    const auto extents = tensor_t_detail::extents_from_tensor<Rank>(tensor);
    auto owner = tensor_t_detail::owner_from_tensor<T>(tensor);
    view_type view(owner.get(), extents_type(extents));
    return TensorT<T, Rank, Access, stdex::layout_right>(std::move(owner), view, access);
  }

  /**
   * @brief Create a legacy Tensor sharing the storage owned by a TensorT view.
   *
   * This bridge is only available for TensorT objects backed by legacy Cytnx storage. The mdspan
   * layout must be exactly representable by Cytnx's current contiguous-or-permuted-contiguous
   * Tensor metadata; otherwise this function throws.
   */
  template <typename T, std::size_t Rank, class Access, class Layout>
  Tensor to_tensor(const TensorT<T, Rank, Access, Layout> &view) {
    if constexpr (Rank == 0) {
      cytnx_error_msg(true, "[ERROR] Cannot convert rank-0 TensorT to legacy Tensor.%s", "\n");
      return Tensor();
    } else {
      auto storage = tensor_t_detail::legacy_storage_from_owner(view.owner());
      cytnx_error_msg(storage->device() != view.device(),
                      "[ERROR] TensorT access device does not match legacy storage device.%s",
                      "\n");

      const auto memory_order = tensor_t_detail::memory_order_from_strides(view);

      std::vector<cytnx_int64> memory_shape;
      memory_shape.reserve(Rank);
      for (const auto axis : memory_order) {
        memory_shape.push_back(static_cast<cytnx_int64>(view.extent(axis)));
      }

      std::vector<cytnx_uint64> perm(Rank);
      for (std::size_t physical_axis = 0; physical_axis < Rank; ++physical_axis) {
        perm[memory_order[physical_axis]] = static_cast<cytnx_uint64>(physical_axis);
      }

      Tensor out = Tensor::from_storage(Storage(storage));
      out = out.reshape(memory_shape);
      out = out.permute(perm);
      return out;
    }
  }

}  // namespace cytnx

#endif  // CYTNX_TENSORT_HPP_
