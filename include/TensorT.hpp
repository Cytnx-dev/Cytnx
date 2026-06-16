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

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <variant>

namespace cytnx {

  class storage_owner {
   public:
    storage_owner() = default;
    explicit storage_owner(boost::intrusive_ptr<Storage_base> storage)
        : storage_(std::move(storage)) {}

    Storage_base *get() const noexcept { return storage_.get(); }
    Storage_base &operator*() const noexcept { return *storage_; }
    Storage_base *operator->() const noexcept { return storage_.get(); }
    explicit operator bool() const noexcept { return static_cast<bool>(storage_); }

    unsigned int dtype() const { return storage_->dtype(); }
    int device() const { return storage_->device(); }
    unsigned long long size() const { return storage_->size(); }

   private:
    boost::intrusive_ptr<Storage_base> storage_;
  };

  template <class T, std::size_t Rank, class Access, class Layout = stdex::layout_stride>
  class TensorT {
   public:
    using element_type = T;
    using access_type = Access;
    using layout_type = Layout;
    using extents_type = stdex::dextents<std::size_t, Rank>;
    using view_type = stdex::mdspan<T, extents_type, Layout>;
    using storage_type = storage_owner;

    static constexpr std::size_t rank() noexcept { return Rank; }

    TensorT() = default;
    TensorT(storage_type storage, view_type view, Access access = Access{})
        : storage_(std::move(storage)), view_(view), access_(access) {}

    std::size_t extent(std::size_t axis) const noexcept { return view_.extent(axis); }
    std::size_t stride(std::size_t axis) const noexcept { return view_.stride(axis); }
    std::size_t required_span_size() const noexcept { return view_.required_span_size(); }

    T *data() const noexcept { return view_.data_handle(); }
    T *data_handle() const noexcept { return view_.data_handle(); }

    const view_type &view() const noexcept { return view_; }
    const storage_type &storage() const noexcept { return storage_; }
    const Access &access() const noexcept { return access_; }

    unsigned int dtype() const { return storage_.dtype(); }
    int device() const { return storage_.device(); }

    template <class... Indices>
    T &operator()(Indices... indices) const noexcept {
      return view_(indices...);
    }

   private:
    storage_type storage_;
    view_type view_;
    [[no_unique_address]] Access access_{};
  };

  template <class T, std::size_t Rank, class Layout = stdex::layout_stride>
  using HostTensorT = TensorT<T, Rank, host_access, Layout>;

  template <class T, class Access, class Layout = stdex::layout_stride>
  using VectorT = TensorT<T, 1, Access, Layout>;

  template <class T, class Access, class Layout = stdex::layout_stride>
  using MatrixT = TensorT<T, 2, Access, Layout>;

  template <class T, std::size_t Rank, class Layout = stdex::layout_stride>
  using TensorDeviceT = std::variant<HostTensorT<T, Rank, Layout>
#ifdef UNI_GPU
                                     ,
                                     TensorT<T, Rank, cuda_access, Layout>
#endif
                                     >;

  namespace tensor_t_detail {

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

    inline storage_owner storage_from_tensor(const Tensor &tensor) {
      return storage_owner(tensor._impl->storage()._impl);
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
    auto storage = tensor_t_detail::storage_from_tensor(tensor);
    view_type view(tensor.ptr_as<T>(), mapping_type(extents_type(extents), strides));
    return TensorT<T, Rank, Access, stdex::layout_stride>(std::move(storage), view, access);
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
    auto storage = tensor_t_detail::storage_from_tensor(tensor);
    view_type view(tensor.ptr_as<T>(), extents_type(extents));
    return TensorT<T, Rank, Access, stdex::layout_right>(std::move(storage), view, access);
  }

}  // namespace cytnx

#endif  // CYTNX_TENSORT_HPP_
