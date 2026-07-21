#include <format>
#include <vector>
#include <map>
#include <random>
#include <string>
#include <type_traits>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/warnings.h>

#include "cytnx.hpp"
#include "pyint_dispatch.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace cytnx;
using pybind_cytnx::dispatch_pyint;

#ifdef BACKEND_TORCH
#else

namespace {
  bool is_empty_tuple(py::handle object) {
    return py::isinstance<py::tuple>(object) &&
           py::reinterpret_borrow<py::tuple>(object).size() == 0;
  }

  void check_tuple_rank(py::tuple args, cytnx_uint64 rank, const char *type_name) {
    const auto index_count = static_cast<cytnx_uint64>(args.size());
    cytnx_error_msg(index_count > rank,
                    "[ERROR] too many indices for %s: got %llu indices for rank-%llu object.%s",
                    type_name, static_cast<unsigned long long>(index_count),
                    static_cast<unsigned long long>(rank), "\n");
  }
}  // namespace

class cHclass {
 public:
  Scalar::Sproxy proxy;

  cHclass(const Scalar::Sproxy &inproxy) { this->proxy = inproxy; }
  cHclass(const cHclass &rhs) { this->proxy = rhs.proxy.copy(); }
  cHclass &operator=(cHclass &rhs) {
    this->proxy = rhs.proxy.copy();
    return *this;
  }

  bool exists() const { return this->proxy.exists(); }
  int dtype() const { return this->proxy._insimpl->dtype(); }

  cytnx_double get_elem_d() const { return cytnx_double(Scalar(this->proxy)); }
  cytnx_float get_elem_f() const { return cytnx_float(Scalar(this->proxy)); }
  cytnx_complex128 get_elem_cd() const { return complex128(Scalar(this->proxy)); }
  cytnx_complex64 get_elem_cf() const { return complex64(Scalar(this->proxy)); }
  cytnx_uint64 get_elem_u64() const { return cytnx_uint64(Scalar(this->proxy)); }
  cytnx_int64 get_elem_i64() const { return cytnx_int64(Scalar(this->proxy)); }
  cytnx_uint32 get_elem_u32() const { return cytnx_uint32(Scalar(this->proxy)); }
  cytnx_int32 get_elem_i32() const { return cytnx_int32(Scalar(this->proxy)); }
  cytnx_uint16 get_elem_u16() const { return cytnx_uint16(Scalar(this->proxy)); }
  cytnx_int16 get_elem_i16() const { return cytnx_int16(Scalar(this->proxy)); }
  cytnx_bool get_elem_b() const { return cytnx_bool(Scalar(this->proxy)); }

  template <class T>
  void set_elem(const T &elem) {
    this->proxy = elem;
  }
};

template <class T>
void f_UniTensor_setelem_scal(UniTensor &self, const std::vector<cytnx_uint64> &locator,
                              const T &rc) {
  self.set_elem(locator, rc);
}

template <class T>
void f_UniTensor_setelem_scal_int(UniTensor &self, const cytnx_uint64 &locator, const T &rc) {
  const std::vector<cytnx_uint64> tmp = {locator};
  self.set_elem(tmp, rc);
}

// Parse UniTensor.get_blocks_ function's silent argument.
//
// This function should be replaced with `py::arg("silent") = false` after stopping
// support for the deprecated typo argument "slient".
inline bool parse_get_blocks_silent_arg(const py::args &args, const py::kwargs &kwargs) {
  bool silent = false;
  if (args.size() + kwargs.size() > 1) {
    throw py::type_error("get_blocks_() takes at most 1 argument");
  }
  if (args.size() == 1) {
    silent = py::cast<bool>(args[0]);
  } else if (kwargs.contains("slient")) {
    py::warnings::warn(
      "Keyword 'slient' is deprecated and will be removed in v2.0.0; use 'silent' instead.",
      PyExc_FutureWarning, 2);
    silent = kwargs["slient"].cast<bool>();
  } else if (kwargs.contains("silent")) {
    silent = kwargs["silent"].cast<bool>();
  } else if (kwargs.size() == 1) {
    // The case that kwargs.size() > 1 has been caught above.
    std::string kwarg_name = py::str(kwargs.begin()->first);
    throw py::type_error(
      std::format("'{}' is an invalid keyword argument for get_blocks_()", kwarg_name));
  }
  return silent;
}

// Phase-2 Task 3 (issue #934, decision record 2026-07-06, amended): the four
// elementwise UniTensor(+)UniTensor operator families split by whether the
// operation is well defined:
//
//   * '+' '-' '+=' '-=' are KEPT but guarded. They require the two operands to
//     describe the same tensor slot -- matching type, rank, labels (in order),
//     rowrank, diagonal-ness, and bonds. With matching metadata the sum /
//     difference is unambiguous and label-preserving; with mismatched metadata
//     the labels would be silently discarded (the #934/#753/#675 complaint), so
//     they raise TypeError instead (see unitensor_addsub_metadata_mismatch).
//   * '*' '/' '*=' '/=' are REMOVED: the elementwise (Hadamard) product /
//     quotient of two UniTensors is basis-dependent and has no tensor-network
//     meaning. Their dunders raise TypeError via raise_unitensor_elementwise_removed.
//
// Scalar<->UniTensor arithmetic (both directions, out-of-place and in-place) is
// unaffected. The C++ operator+/operator- are unchanged -- the Krylov solvers
// rely on them as vector-space operations.
inline constexpr const char *kUniTensorMulRemovedGuidance =
  "Use Contract() for tensor contraction or Kron() for tensor (outer) products; for a "
  "genuinely elementwise (Hadamard) product, operate on the raw blocks instead -- "
  "Tensor-level elementwise arithmetic is fully supported via ut.get_block(). The Hadamard "
  "product of two UniTensors is basis-dependent and not a tensor-network operation. Scalar "
  "multiplication ('2.0 * ut', 'ut *= 2.0') is unaffected.";

inline constexpr const char *kUniTensorDivRemovedGuidance =
  "Use Contract() with an inverted operator, or scalar division ('ut / 2.0', 'ut /= 2.0') "
  "if that is what you meant; for a genuinely elementwise division, operate on the raw "
  "blocks instead -- Tensor-level elementwise arithmetic is fully supported via "
  "ut.get_block(). Division of two UniTensors has no well-defined tensor-network meaning.";

[[noreturn]] inline void raise_unitensor_elementwise_removed(const std::string &op_name,
                                                             const std::string &alt) {
  throw py::type_error(std::format(
    "elementwise UniTensor{}UniTensor arithmetic was removed (issue #934): {}", op_name, alt));
}

// Render a label list for diagnostics, e.g. ['a', 'b'].
inline std::string format_labels(const std::vector<std::string> &labels) {
  std::string s = "[";
  for (std::size_t i = 0; i < labels.size(); ++i) {
    if (i) s += ", ";
    s += "'" + labels[i] + "'";
  }
  s += "]";
  return s;
}

// Returns "" when a and b have matching metadata for elementwise +/-, otherwise a
// human-readable reason for the mismatch. +/- is only well-defined (and label-preserving)
// when both operands describe the same tensor slot.
inline std::string unitensor_addsub_metadata_mismatch(const UniTensor &a, const UniTensor &b) {
  if (a.uten_type() != b.uten_type())
    return std::format("different UniTensor types ({} vs {})", a.uten_type_str(),
                       b.uten_type_str());
  if (a.rank() != b.rank()) return std::format("different rank ({} vs {})", a.rank(), b.rank());
  if (a.labels() != b.labels())
    return std::format("different labels ({} vs {})", format_labels(a.labels()),
                       format_labels(b.labels()));
  if (a.rowrank() != b.rowrank())
    return std::format("different rowrank ({} vs {})", a.rowrank(), b.rowrank());
  if (a.is_diag() != b.is_diag())
    return std::string("one is diagonal (is_diag=True) and the other is not");
  const std::vector<Bond> &ba = a.bonds();
  const std::vector<Bond> &bb = b.bonds();
  for (std::size_t i = 0; i < ba.size(); ++i) {
    if (ba[i] != bb[i])
      return std::format("bond at leg {} (label '{}') does not match", i, a.labels()[i]);
  }
  return std::string();
}

[[noreturn]] inline void raise_unitensor_addsub_metadata_mismatch(const std::string &op_name,
                                                                  const std::string &reason) {
  throw py::type_error(std::format(
    "UniTensor{}UniTensor requires the two operands to have matching metadata, but they have {}. "
    "Elementwise +/- is only defined when both describe the same tensor slot (same labels, bonds, "
    "rowrank, ...); align them first (e.g. permute()/relabel_()), or use Contract() / operate on "
    "ut.get_block() for genuinely elementwise math. See issue #934.",
    op_name, reason));
}

// Lambda used for _getitem__ and _setitem__
auto build_accessors = [](const UniTensor &self, py::object locators) {
  ssize_t start, stop, step, slicelength;
  std::vector<cytnx::Accessor> accessors;
  if (self.is_diag()) {
    if (py::isinstance<py::tuple>(locators)) {
      py::tuple Args = locators.cast<py::tuple>();
      check_tuple_rank(Args, self.rank(), "UniTensor");
      cytnx_error_msg(Args.size() > 2,
                      "[ERROR][slicing] A diagonal UniTensor can only be accessed with one- or "
                      "two dimensional slicing.%s",
                      "\n");
      // mixing of slice and ints
      for (cytnx_uint32 axis = 0; axis < Args.size(); axis++) {
        // check type:
        if (py::isinstance<py::slice>(Args[axis])) {
          py::slice sls = Args[axis].cast<py::slice>();
          if (!sls.compute((ssize_t)self.shape()[axis], &start, &stop, &step, &slicelength))
            throw py::error_already_set();
          accessors.push_back(
            cytnx::Accessor::range(cytnx_int64(start), cytnx_int64(stop), cytnx_int64(step)));
        } else {
          accessors.push_back(cytnx::Accessor(Args[axis].cast<cytnx_int64>()));
        }
      }
    } else if (py::isinstance<py::slice>(locators)) {
      py::slice sls = locators.cast<py::slice>();
      if (!sls.compute((ssize_t)self.shape()[0], &start, &stop, &step, &slicelength))
        throw py::error_already_set();
      accessors.push_back(cytnx::Accessor::range(start, stop, step));
    } else {
      accessors.push_back(cytnx::Accessor(locators.cast<cytnx_int64>()));
    }
  } else {
    if (py::isinstance<py::tuple>(locators)) {
      py::tuple Args = locators.cast<py::tuple>();
      check_tuple_rank(Args, self.rank(), "UniTensor");
      cytnx_uint64 cnt = 0;
      // mixing of slice and ints
      for (cytnx_uint32 axis = 0; axis < Args.size(); axis++) {
        cnt++;
        // check type:
        if (py::isinstance<py::slice>(Args[axis])) {
          py::slice sls = Args[axis].cast<py::slice>();
          if (!sls.compute((ssize_t)self.shape()[axis], &start, &stop, &step, &slicelength))
            throw py::error_already_set();
          // if(slicelength == self.shape()[axis])
          // accessors.push_back(cytnx::Accessor::all());
          accessors.push_back(
            cytnx::Accessor::range(cytnx_int64(start), cytnx_int64(stop), cytnx_int64(step)));
        } else {
          accessors.push_back(cytnx::Accessor(Args[axis].cast<cytnx_int64>()));
        }
      }
      while (cnt < self.shape().size()) {
        cnt++;
        accessors.push_back(Accessor::all());
      }
    } else if (py::isinstance<py::slice>(locators)) {
      py::slice sls = locators.cast<py::slice>();
      if (!sls.compute((ssize_t)self.shape()[0], &start, &stop, &step, &slicelength))
        throw py::error_already_set();
      // if(slicelength == self.shape()[0]) accessors.push_back(cytnx::Accessor::all());
      accessors.push_back(cytnx::Accessor::range(start, stop, step));
      for (cytnx_uint32 axis = 1; axis < self.shape().size(); axis++) {
        accessors.push_back(Accessor::all());
      }
    } else {
      // only int
      for (cytnx_uint32 i = 0; i < self.shape().size(); i++) {
        if (i == 0)
          accessors.push_back(cytnx::Accessor(locators.cast<cytnx_int64>()));
        else
          accessors.push_back(cytnx::Accessor::all());
      }
    }
  }
  return accessors;
};

namespace {
  // Deprecated combineBonds()/Norm() binding helpers. Calling the [[deprecated]]
  // UniTensor::combineBonds / UniTensor::Norm warns; the suppression lives here at file scope
  // because a #pragma GCC diagnostic cannot legally sit inside the .def() chain expression under
  // GCC (only clang tolerated that). These bindings intentionally keep exposing the deprecated
  // calls for one release: use combineBond_()/combineBond() and norm() instead.
  #if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  #endif
  py::object unitensor_combineBonds_int_deprecated(py::object self,
                                                   const std::vector<cytnx_int64> &indicators,
                                                   const bool &force, const bool &by_label) {
    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "combineBonds() is deprecated, use combineBond_()/combineBond() instead.",
                     1) < 0)
      throw py::error_already_set();
    auto &self_ref = self.cast<UniTensor &>();
    if (by_label) {
      cytnx_warning_msg(true,
                        "[Deprecated notice] by_label option is going to be deprecated. using "
                        "string will automatically recognized as labels.%s",
                        "\n");
      self_ref.combineBonds(indicators, force, by_label);
    } else {
      self_ref.combineBonds(indicators, force);
    }
    return self;
  }
  py::object unitensor_combineBonds_str_deprecated(py::object self,
                                                   const std::vector<std::string> &indicators,
                                                   const bool &force) {
    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "combineBonds() is deprecated, use combineBond_()/combineBond() instead.",
                     1) < 0)
      throw py::error_already_set();
    self.cast<UniTensor &>().combineBonds(indicators, force);
    return self;
  }
  Tensor unitensor_Norm_deprecated(UniTensor &self) {
    if (PyErr_WarnEx(PyExc_DeprecationWarning, "Norm() is deprecated, use norm() instead.", 1) < 0)
      throw py::error_already_set();
    return self.Norm();
  }
  #if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic pop
  #endif
  // Converts a homogeneous list of numpy integer scalars to vector<cytnx_int64>
  // (or vector<cytnx_uint64> for numpy_scalar<uint64_t>, the one numpy width
  // that doesn't always fit in int64), then calls fn with it. numpy integer
  // scalars are not subclasses of Python int, so they don't match
  // dispatch_pyint_vector's std::vector<py::int_> parameter below -- without
  // this overload, get_block/get_block_'s qnum parameter would silently stop
  // accepting np.int64/np.int32/etc. elements it accepted before this
  // consolidation (when it had raw vector<cytnx_int64>/vector<cytnx_uint64>
  // overloads, whose arithmetic-type casters accept anything with __index__).
  template <class T, class Fn>
  auto dispatch_numpy_int_vector(const std::vector<py::numpy_scalar<T>> &vals, Fn &&fn) {
    if constexpr (std::is_same_v<T, uint64_t>) {
      std::vector<cytnx_uint64> converted;
      converted.reserve(vals.size());
      for (const auto &v : vals) converted.push_back(static_cast<cytnx_uint64>(static_cast<T>(v)));
      return fn(converted);
    } else {
      std::vector<cytnx_int64> converted;
      converted.reserve(vals.size());
      for (const auto &v : vals) converted.push_back(static_cast<cytnx_int64>(static_cast<T>(v)));
      return fn(converted);
    }
  }

  // Converts a list of Python ints to a single homogeneous vector<cytnx_int64>
  // or vector<cytnx_uint64> (int64 unless any element needs uint64's range),
  // then calls fn with it. Vector counterpart of dispatch_pyint
  // (pyint_dispatch.hpp).
  template <class Fn>
  auto dispatch_pyint_vector(const std::vector<py::int_> &vals, Fn &&fn) {
    bool needs_uint64 = false;
    for (const py::int_ &v : vals) {
      int overflow = 0;
      PyLong_AsLongLongAndOverflow(v.ptr(), &overflow);
      if (overflow < 0) {
        cytnx_error_msg(true, "[ERROR] integer scalar out of the supported int64/uint64 range.%s",
                        "\n");
      } else if (overflow > 0) {
        needs_uint64 = true;
      }
    }
    if (needs_uint64) {
      std::vector<cytnx_uint64> converted;
      converted.reserve(vals.size());
      for (const py::int_ &v : vals) converted.push_back(v.cast<cytnx_uint64>());
      return fn(converted);
    }
    std::vector<cytnx_int64> converted;
    converted.reserve(vals.size());
    for (const py::int_ &v : vals) converted.push_back(v.cast<cytnx_int64>());
    return fn(converted);
  }

}  // namespace

void unitensor_binding(py::module &m) {
  py::class_<cHclass>(m, "Helpclass")
    .def("exists", &cHclass::exists)
    .def("dtype", &cHclass::dtype)
    .def("get_elem_d", &cHclass::get_elem_d)
    .def("get_elem_f", &cHclass::get_elem_f)
    .def("get_elem_cd", &cHclass::get_elem_cd)
    .def("get_elem_cf", &cHclass::get_elem_cf)
    .def("get_elem_i64", &cHclass::get_elem_i64)
    .def("get_elem_u64", &cHclass::get_elem_u64)
    .def("get_elem_i32", &cHclass::get_elem_i32)
    .def("get_elem_u32", &cHclass::get_elem_u32)
    .def("get_elem_i16", &cHclass::get_elem_i16)
    .def("get_elem_u16", &cHclass::get_elem_u16)
    .def("get_elem_b", &cHclass::get_elem_b)

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("set_elem",
         [](cHclass &self, const py::numpy_scalar<float> &rhs) {
           self.set_elem(static_cast<cytnx_float>(rhs));
         })
    .def("set_elem",
         [](cHclass &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           self.set_elem(static_cast<cytnx_complex64>(rhs));
         })
    .def("set_elem",
         [](cHclass &self, const py::numpy_scalar<int64_t> &rhs) {
           self.set_elem(static_cast<cytnx_int64>(rhs));
         })
    .def("set_elem",
         [](cHclass &self, const py::numpy_scalar<uint64_t> &rhs) {
           self.set_elem(static_cast<cytnx_uint64>(rhs));
         })
    .def("set_elem",
         [](cHclass &self, const py::numpy_scalar<int32_t> &rhs) {
           self.set_elem(static_cast<cytnx_int32>(rhs));
         })
    .def("set_elem",
         [](cHclass &self, const py::numpy_scalar<uint32_t> &rhs) {
           self.set_elem(static_cast<cytnx_uint32>(rhs));
         })
    .def("set_elem",
         [](cHclass &self, const py::numpy_scalar<int16_t> &rhs) {
           self.set_elem(static_cast<cytnx_int16>(rhs));
         })
    .def("set_elem",
         [](cHclass &self, const py::numpy_scalar<uint16_t> &rhs) {
           self.set_elem(static_cast<cytnx_uint16>(rhs));
         })
    .def("set_elem",
         [](cHclass &self, const py::numpy_scalar<bool> &rhs) {
           self.set_elem(static_cast<cytnx_bool>(rhs));
         })
    .def("set_elem",
         [](cHclass &self, const py::int_ &rhs) {
           dispatch_pyint(rhs, [&](auto v) { self.set_elem(v); });
         })
    .def("set_elem", [](cHclass &self, const cytnx_double &rhs) { self.set_elem(rhs); })
    .def("set_elem", [](cHclass &self, const cytnx_complex128 &rhs) { self.set_elem(rhs); });

  // entry.UniTensor
  py::class_<UniTensor>(m, "UniTensor")
    .def(py::init<>())
    .def(py::init<const cytnx::Tensor &, const bool &, const cytnx_int64 &, const std::vector<std::string> &, const std::string &>(), py::arg("Tin"),
         py::arg("is_diag") = false, py::arg("rowrank") = (cytnx_int64)(-1), py::arg("labels") = std::vector<std::string>(), py::arg("name")="")


    .def(py::init<const std::vector<Bond> &, const std::vector<std::string> &, const cytnx_int64 &,
                  const unsigned int &, const int &, const bool &, const std::string &>(),
         py::arg("bonds"), py::arg("labels") = std::vector<std::string>(),
         py::arg("rowrank") = (cytnx_int64)(-1),
         py::arg("dtype") = (unsigned int)(cytnx::Type.Double),
         py::arg("device") = (int)cytnx::Device.cpu, py::arg("is_diag") = false, py::arg("name")="")


    .def("Init",[](UniTensor &self, const Tensor &in_tensor, const bool &is_diag, const cytnx_int64 &rowrank, const std::vector<std::string> &labels, const std::string &name){
                    self.Init(in_tensor,is_diag,rowrank,labels,name);
                },py::arg("Tin"),py::arg("is_diag")=false,py::arg("rowrank")=(cytnx_int64)(-1), py::arg("labels") = std::vector<std::string>(), py::arg("name")="")


    .def("Init",[](UniTensor &self, const std::vector<Bond> &bonds, const std::vector<std::string> &in_labels,
                   const cytnx_int64 &rowrank, const unsigned int &dtype,
                   const int &device, const bool &is_diag, const std::string &name){
                    self.Init(bonds,in_labels,rowrank,dtype,device,is_diag,name);
                },
         py::arg("bonds"), py::arg("labels") = std::vector<std::string>(),
         py::arg("rowrank") = (cytnx_int64)(-1),
         py::arg("dtype") = (unsigned int)(cytnx::Type.Double),
         py::arg("device") = (int)cytnx::Device.cpu, py::arg("is_diag") = false, py::arg("name")="")
    .def("set_name_",
         [](py::object self, const std::string &name) {
           self.cast<UniTensor &>().set_name_(name);
           return self;
         })
    .def("set_name",
         [](py::object self, const std::string &name) {
           if (PyErr_WarnEx(PyExc_DeprecationWarning,
             "set_name() is deprecated, use set_name_() instead.", 1) < 0) throw py::error_already_set();
           self.cast<UniTensor &>().set_name_(name);
           return self;
         })


    .def("set_label_",
         [](py::object self, const cytnx_int64 &idx, const std::string &new_label) {
           self.cast<UniTensor &>().set_label_(idx, new_label);
           return self;
         }, py::arg("idx"), py::arg("new_label"))

    .def("set_label_",
         [](py::object self, const std::string &old_label, const std::string &new_label) {
           self.cast<UniTensor &>().set_label_(old_label, new_label);
           return self;
         }, py::arg("old_label"), py::arg("new_label"))

    .def("set_label",
         [](py::object self, const cytnx_int64 &idx, const std::string &new_label) {
           if (PyErr_WarnEx(PyExc_DeprecationWarning,
             "set_label() is deprecated, use set_label_() instead.", 1) < 0) throw py::error_already_set();
           self.cast<UniTensor &>().set_label_(idx, new_label);
           return self;
         }, py::arg("idx"), py::arg("new_label"))

    .def("set_label",
         [](py::object self, const std::string &old_label, const std::string &new_label) {
           if (PyErr_WarnEx(PyExc_DeprecationWarning,
             "set_label() is deprecated, use set_label_() instead.", 1) < 0) throw py::error_already_set();
           self.cast<UniTensor &>().set_label_(old_label, new_label);
           return self;
         }, py::arg("old_label"), py::arg("new_label"))


    .def("set_labels",
         [](py::object self, const std::vector<std::string> &new_labels) {
           if (PyErr_WarnEx(PyExc_DeprecationWarning,
             "set_labels() is deprecated, use relabel_() instead.", 1) < 0) throw py::error_already_set();
           self.cast<UniTensor &>().relabel_(new_labels);
           return self;
         }, py::arg("new_labels"))


    .def("set_rowrank_",
         [](py::object self, const cytnx_uint64 &new_rowrank) {
           self.cast<UniTensor &>().set_rowrank_(new_rowrank);
           return self;
         }, py::arg("new_rowrank"))

    .def("set_rowrank", &UniTensor::set_rowrank, py::arg("new_rowrank"))
    .def("relabel",[](UniTensor &self, const std::vector<std::string> &new_labels){
                        return self.relabel(new_labels);
                    }, py::arg("new_labels"))
    .def("relabels",[](UniTensor &self, const std::vector<std::string> &new_labels){
                        if (PyErr_WarnEx(PyExc_DeprecationWarning,
                          "relabels() is deprecated, use relabel() instead.", 1) < 0) throw py::error_already_set();
                        return self.relabel(new_labels);
                    }, py::arg("new_labels"))

     .def("relabel_",[](py::object self, const std::vector<std::string> &new_labels){
                        self.cast<UniTensor &>().relabel_(new_labels);
                        return self;
                    }, py::arg("new_labels"))
     .def("relabels_",[](py::object self, const std::vector<std::string> &new_labels){
                        if (PyErr_WarnEx(PyExc_DeprecationWarning,
                          "relabels_() is deprecated, use relabel_() instead.", 1) < 0) throw py::error_already_set();
                        self.cast<UniTensor &>().relabel_(new_labels);
                        return self;
                    }, py::arg("new_labels"))


    .def("relabel", [](UniTensor &self, const cytnx_int64 &idx, const std::string &new_label){
                            return self.relabel(idx,new_label);
                        },py::arg("idx"), py::arg("new_label"))

     .def("relabel_", [](py::object self, const cytnx_int64 &idx, const std::string &new_label){
                            self.cast<UniTensor &>().relabel_(idx,new_label);
                            return self;
                        },py::arg("idx"), py::arg("new_label"))

    .def("relabel", [](UniTensor &self, const std::string &old_label, const std::string &new_label){
                            return self.relabel(old_label,new_label);
                        },py::arg("old_label"), py::arg("new_label"))
     .def("relabel_", [](py::object self, const std::string &old_label, const std::string &new_label){
                            self.cast<UniTensor &>().relabel_(old_label,new_label);
                            return self;
                        },py::arg("old_label"), py::arg("new_label"))


    .def("relabel",[](UniTensor &self, const std::vector<std::string> &old_labels, const std::vector<std::string> &new_labels){
                        return self.relabel(old_labels,new_labels);
                    } ,py::arg("old_labels"), py::arg("new_labels"))

    .def("relabel_",[](py::object self, const std::vector<std::string> &old_labels, const std::vector<std::string> &new_labels){
                        self.cast<UniTensor &>().relabel_(old_labels,new_labels);
                        return self;
                    } ,py::arg("old_labels"), py::arg("new_labels"))

    .def("relabels",[](UniTensor &self, const std::vector<std::string> &old_labels, const std::vector<std::string> &new_labels){
                        if (PyErr_WarnEx(PyExc_DeprecationWarning,
                          "relabels() is deprecated, use relabel() instead.", 1) < 0) throw py::error_already_set();
                        return self.relabel(old_labels,new_labels);
                    } ,py::arg("old_labels"), py::arg("new_labels"))

    .def("relabels_",[](py::object self, const std::vector<std::string> &old_labels, const std::vector<std::string> &new_labels){
                        if (PyErr_WarnEx(PyExc_DeprecationWarning,
                          "relabels_() is deprecated, use relabel_() instead.", 1) < 0) throw py::error_already_set();
                        self.cast<UniTensor &>().relabel_(old_labels,new_labels);
                        return self;
                    } ,py::arg("old_labels"), py::arg("new_labels"))



    .def("rowrank", &UniTensor::rowrank)
    .def("Nblocks", &UniTensor::Nblocks)
    .def("rank", &UniTensor::rank)
    .def("size", &UniTensor::size)
    .def("uten_type", &UniTensor::uten_type)
    .def("uten_type_str",&UniTensor::uten_type_str)
    .def("is_void", &UniTensor::is_void)
    .def("is_scalar", &UniTensor::is_scalar)
    .def("is_empty", &UniTensor::is_empty)
    .def("syms", &UniTensor::syms)
    .def("dtype", &UniTensor::dtype)
    .def("dtype_str", &UniTensor::dtype_str)
    .def("device", &UniTensor::device)
    .def("device_str", &UniTensor::device_str)
    .def("name", &UniTensor::name)
    .def("is_blockform", &UniTensor::is_blockform)

    .def("get_index",&UniTensor::get_index)

    // reshape()/reshape_() accept both the variadic form (ut.reshape(2, 3)) and the
    // list form (ut.reshape([2, 3])); see pybind_cytnx::parse_index_args in
    // pyint_dispatch.hpp (#293, ruling 4).
    .def("reshape",
         [](UniTensor &self, py::args args, py::kwargs kwargs) -> UniTensor {
           std::vector<cytnx::cytnx_int64> c_args =
             pybind_cytnx::parse_index_args<cytnx::cytnx_int64>(args);
           cytnx_uint64 rowrank = 0;

           if (kwargs) {
             if (kwargs.contains("rowrank")) rowrank = kwargs["rowrank"].cast<cytnx::cytnx_int64>();
           }

           return self.reshape(c_args, rowrank);
         })
    .def("reshape_",
         [](UniTensor &self, py::args args, py::kwargs kwargs) {
           std::vector<cytnx::cytnx_int64> c_args =
             pybind_cytnx::parse_index_args<cytnx::cytnx_int64>(args);
           cytnx_uint64 rowrank = 0;

           if (kwargs) {
             if (kwargs.contains("rowrank")) rowrank = kwargs["rowrank"].cast<cytnx::cytnx_int64>();
           }

           return &self.reshape_(c_args, rowrank);
         })
    .def("elem_exists", &UniTensor::elem_exists)
    .def("item",
         [](UniTensor &self) {
           py::object out;
           if (self.dtype() == cytnx::Type.Double)
             out = py::cast(self.item<cytnx::cytnx_double>());
           else if (self.dtype() == cytnx::Type.Float)
             out = py::cast(self.item<cytnx::cytnx_float>());
           else if (self.dtype() == cytnx::Type.ComplexDouble)
             out = py::cast(self.item<cytnx::cytnx_complex128>());
           else if (self.dtype() == cytnx::Type.ComplexFloat)
             out = py::cast(self.item<cytnx::cytnx_complex64>());
           else if (self.dtype() == cytnx::Type.Uint64)
             out = py::cast(self.item<cytnx::cytnx_uint64>());
           else if (self.dtype() == cytnx::Type.Int64)
             out = py::cast(self.item<cytnx::cytnx_int64>());
           else if (self.dtype() == cytnx::Type.Uint32)
             out = py::cast(self.item<cytnx::cytnx_uint32>());
           else if (self.dtype() == cytnx::Type.Int32)
             out = py::cast(self.item<cytnx::cytnx_int32>());
           else if (self.dtype() == cytnx::Type.Uint16)
             out = py::cast(self.item<cytnx::cytnx_uint16>());
           else if (self.dtype() == cytnx::Type.Int16)
             out = py::cast(self.item<cytnx::cytnx_int16>());
           else if (self.dtype() == cytnx::Type.Bool)
             out = py::cast(self.item<cytnx::cytnx_bool>());
           else
             cytnx_error_msg(true, "%s", "[ERROR] try to get element from a empty UniTensor.");
           return out;
         })

    .def("c_at", [](UniTensor &self, const std::vector<cytnx_uint64> &locator){
                  Scalar::Sproxy tmp = self.at(locator);
                  return cHclass(tmp);
               },py::arg("locator"))


    .def("c_at",[](UniTensor &self, const std::vector<std::string> &labels, const std::vector<cytnx_uint64> &locator){
                  Scalar::Sproxy tmp = self.at(labels,locator);
                  return cHclass(tmp);
               },py::arg("labels"), py::arg("locator"))

     .def("__getitem__",
         [](const UniTensor &self, py::object locators) {
           cytnx_error_msg(self.uten_type() == UTenType.Void,
                           "[ERROR] try to getitem from an uninitialized UniTensor%s", "\n");
           cytnx_error_msg(
             self.uten_type() != UTenType.Dense,
             "[ERROR] Cannot get element using [] from Block/BlockFermionicUniTensor. Use at() instead.%s", "\n");
           if (self.rank() == 0) {
             cytnx_error_msg(!is_empty_tuple(locators),
                             "[ERROR] rank-0 UniTensor can only be indexed with ().%s", "\n");
             return self.get(std::vector<Accessor>{});
           }

           auto accessors = build_accessors(self, locators);
           return self.get(accessors);
         })
    .def("__setitem__",
         [](UniTensor &self, py::object locators, const cytnx::Tensor &rhs) {
           cytnx_error_msg(self.uten_type() == UTenType.Void,
                           "[ERROR] try to setelem to an uninitialized UniTensor%s", "\n");
           cytnx_error_msg(self.uten_type() == UTenType.Sparse, "[ERROR] SparseUniTensor is deprecated. Use BlockUniTensor or LinOp instead.%s", "\n");
           if (self.rank() == 0) {
             cytnx_error_msg(!is_empty_tuple(locators),
                             "[ERROR] rank-0 UniTensor can only be indexed with ().%s", "\n");
             self.set(std::vector<Accessor>{}, rhs);
             return;
           }

           auto accessors = build_accessors(self, locators);
           self.set(accessors, rhs);
         })
    .def("__setitem__",
         [](UniTensor &self, py::object locators, const cytnx::UniTensor &rhs) {
           cytnx_error_msg(self.uten_type() == UTenType.Void,
                           "[ERROR] try to setelem to an uninitialized UniTensor%s", "\n");
           cytnx_error_msg(
             self.uten_type() != UTenType.Dense,
             "[ERROR] cannot set element using [] from Block/BlockFermionicUniTensor. Use at() instead.%s", "\n");
           if (self.rank() == 0) {
             cytnx_error_msg(!is_empty_tuple(locators),
                             "[ERROR] rank-0 UniTensor can only be indexed with ().%s", "\n");
             self.set(std::vector<Accessor>{}, rhs.get_block());
             return;
           }

           auto accessors = build_accessors(self, locators);
           self.set(accessors, rhs.get_block());
         })

    .def("get_elem",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator) {
           py::object out;
           if (self.dtype() == cytnx::Type.Double)
             out = py::cast(self.get_elem<cytnx::cytnx_double>(locator));
           else if (self.dtype() == cytnx::Type.Float)
             out = py::cast(self.get_elem<cytnx::cytnx_float>(locator));
           else if (self.dtype() == cytnx::Type.ComplexDouble)
             out = py::cast(self.get_elem<cytnx::cytnx_complex128>(locator));
           else if (self.dtype() == cytnx::Type.ComplexFloat)
             out = py::cast(self.get_elem<cytnx::cytnx_complex64>(locator));
           else
             cytnx_error_msg(true, "%s", "[ERROR] try to get element from a void Storage.");
           return out;
         })

    // set_elem/__setitem__ keep-set; registration ORDER matters -- see
    // "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp. Group-specific note
    // (scope extension beyond the plan's literal Tensor-only inventory):
    // recon found UniTensor's set_elem/__setitem__ had the SAME
    // registration-order bug as Tensor's __setitem__ -- np.float32 was
    // accepted by the FIRST-registered cytnx_complex128 overload (the
    // __float__-fallback trap), so `ut[0] = np.float32(2.5)` on a Float
    // UniTensor did not merely mis-preserve dtype, it raised a hard
    // RuntimeError ("Cannot set Complex Double to Float"). Fixed the same
    // way for both the vector-locator (explicit index list) and
    // single-int-locator (diagonal UniTensor) overload families.
    .def("set_elem",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<float> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_float>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<std::complex<float>> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_complex64>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<int64_t> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_int64>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<uint64_t> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_uint64>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<int32_t> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_int32>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<uint32_t> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_uint32>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<int16_t> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_int16>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<uint16_t> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_uint16>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<bool> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_bool>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator, const py::int_ &rc) {
           dispatch_pyint(rc, [&](auto v) { f_UniTensor_setelem_scal(self, locator, v); });
         })
    .def("set_elem", &f_UniTensor_setelem_scal<cytnx_double>)
    .def("set_elem", &f_UniTensor_setelem_scal<cytnx_complex128>)

    .def("set_elem",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::numpy_scalar<float> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_float>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const cytnx_uint64 &locator,
            const py::numpy_scalar<std::complex<float>> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_complex64>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::numpy_scalar<int64_t> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_int64>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::numpy_scalar<uint64_t> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_uint64>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::numpy_scalar<int32_t> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_int32>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::numpy_scalar<uint32_t> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_uint32>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::numpy_scalar<int16_t> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_int16>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::numpy_scalar<uint16_t> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_uint16>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::numpy_scalar<bool> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_bool>(rc));
         })
    .def("set_elem",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::int_ &rc) {
           dispatch_pyint(rc, [&](auto v) { f_UniTensor_setelem_scal_int(self, locator, v); });
         })
    .def("set_elem", &f_UniTensor_setelem_scal_int<cytnx_double>)
    .def("set_elem", &f_UniTensor_setelem_scal_int<cytnx_complex128>)

    .def("__setitem__",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<float> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_float>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<std::complex<float>> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_complex64>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<int64_t> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_int64>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<uint64_t> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_uint64>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<int32_t> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_int32>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<uint32_t> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_uint32>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<int16_t> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_int16>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<uint16_t> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_uint16>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator,
            const py::numpy_scalar<bool> &rc) {
           f_UniTensor_setelem_scal(self, locator, static_cast<cytnx_bool>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator, const py::int_ &rc) {
           dispatch_pyint(rc, [&](auto v) { f_UniTensor_setelem_scal(self, locator, v); });
         })
    .def("__setitem__", &f_UniTensor_setelem_scal<cytnx_double>)
    .def("__setitem__", &f_UniTensor_setelem_scal<cytnx_complex128>)

    .def("__setitem__",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::numpy_scalar<float> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_float>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const cytnx_uint64 &locator,
            const py::numpy_scalar<std::complex<float>> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_complex64>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::numpy_scalar<int64_t> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_int64>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::numpy_scalar<uint64_t> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_uint64>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::numpy_scalar<int32_t> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_int32>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::numpy_scalar<uint32_t> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_uint32>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::numpy_scalar<int16_t> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_int16>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::numpy_scalar<uint16_t> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_uint16>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::numpy_scalar<bool> &rc) {
           f_UniTensor_setelem_scal_int(self, locator, static_cast<cytnx_bool>(rc));
         })
    .def("__setitem__",
         [](UniTensor &self, const cytnx_uint64 &locator, const py::int_ &rc) {
           dispatch_pyint(rc, [&](auto v) { f_UniTensor_setelem_scal_int(self, locator, v); });
         })
    .def("__setitem__", &f_UniTensor_setelem_scal_int<cytnx_double>)
    .def("__setitem__", &f_UniTensor_setelem_scal_int<cytnx_complex128>)

    .def("is_contiguous", &UniTensor::is_contiguous)
    .def("is_diag", &UniTensor::is_diag)
    .def("is_tag", &UniTensor::is_tag)
    .def("is_braket_form", &UniTensor::is_braket_form)
    .def("same_data", &UniTensor::same_data)
    .def("labels", &UniTensor::labels)
    .def("bonds", [](UniTensor &self) { return self.bonds(); })
    .def("bond_", [](UniTensor &self, const cytnx_uint64 &idx){return self.bond_(idx);} ,py::arg("idx"))
    .def("bond_", [](UniTensor &self, const std::string &label){return self.bond_(label);} ,py::arg("label"))
    .def("bond", [](UniTensor &self, const cytnx_uint64 &idx){return self.bond(idx);} ,py::arg("idx"))
    .def("bond", [](UniTensor &self, const std::string &label){return self.bond(label);} ,py::arg("label"))
    .def("shape", &UniTensor::shape)
    .def("signflip", &UniTensor::signflip)
    .def("to_", &UniTensor::to_)
    .def(
      "to_different_device",
      [](UniTensor &self, const cytnx_int64 &device) {
        cytnx_error_msg(self.device() == device,
                        "[ERROR][pybind][to_diffferent_device] same device for to() should be "
                        "handled on the Python side.%s",
                        "\n");
        return self.to(device);
      },
      py::arg("device"))
    .def("clone", &UniTensor::clone)
    .def("__copy__", &UniTensor::clone)
    .def("__deepcopy__", &UniTensor::clone)
    .def(
      "Save", [](UniTensor &self, const std::string &fname) { self.Save(fname); }, py::arg("fname"))
    .def_static(
      "Load", [](const std::string &fname) { return UniTensor::Load(fname); }, py::arg("fname"))
    //.def("permute",&UniTensor::permute,py::arg("mapper"),py::arg("rowrank")=(cytnx_int64)-1,py::arg("by_label")=false)
    //.def("permute_",&UniTensor::permute_,py::arg("mapper"),py::arg("rowrank")=(cytnx_int64)-1,py::arg("by_label")=false)
    .def(
      "astype_different_type",
      [](cytnx::UniTensor &self, const cytnx_uint64 &new_type) {
        cytnx_error_msg(self.dtype() == new_type,
                        "[ERROR][pybind][astype_diffferent_type] same type for astype() should be "
                        "handle in python side.%s",
                        "\n");
        return self.astype(new_type);
      },
      py::arg("new_type"))

    // [Deprecated by_label!]
    .def("permute_", [](UniTensor &self, const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank){
                        return &self.permute_(mapper,rowrank);
                },py::arg("mapper"), py::arg("rowrank")=(cytnx_int64)(-1))

    .def("permute_", [](UniTensor &self, const std::vector<std::string> &mapper, const cytnx_int64 &rowrank){
                        return &self.permute_(mapper,rowrank);
                },py::arg("mapper"), py::arg("rowrank")=(cytnx_int64)(-1))

    .def("permute", [](UniTensor &self, const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank){
                        return self.permute(mapper,rowrank);
                },py::arg("mapper"), py::arg("rowrank")=(cytnx_int64)(-1))

    .def("permute", [](UniTensor &self, const std::vector<std::string> &mapper, const cytnx_int64 &rowrank){
                        return self.permute(mapper,rowrank);
                },py::arg("mapper"), py::arg("rowrank")=(cytnx_int64)(-1))

    // permute_()/permute() ALSO accept the variadic forms (ut.permute_(1, 2, 0) and
    // ut.permute_("a", "b", "c")), on top of the list forms bound just above (#293,
    // ruling 4). These py::args catch-all overloads are registered AFTER the
    // specific std::vector<...> overloads so a genuine single-list call
    // (ut.permute([1, 2, 0])) keeps matching those first; only calls with 0, 2+
    // positional args, or exactly 1 non-list/tuple positional arg fall through to
    // here. is_string_args()/parse_index_args() (pyint_dispatch.hpp) resolve
    // whether the mapper is int- or string-valued, and unwrap a sole list/tuple
    // argument if present so both variadic and list calls funnel through the same
    // path once they reach this overload.
    .def("permute_",
         [](UniTensor &self, py::args args, py::kwargs kwargs) {
           cytnx_int64 rowrank = -1;
           if (kwargs && kwargs.contains("rowrank")) rowrank = kwargs["rowrank"].cast<cytnx_int64>();
           if (pybind_cytnx::is_string_args(args)) {
             return &self.permute_(pybind_cytnx::parse_index_args<std::string>(args), rowrank);
           }
           return &self.permute_(pybind_cytnx::parse_index_args<cytnx_int64>(args), rowrank);
         })

    .def("permute",
         [](UniTensor &self, py::args args, py::kwargs kwargs) -> UniTensor {
           cytnx_int64 rowrank = -1;
           if (kwargs && kwargs.contains("rowrank")) rowrank = kwargs["rowrank"].cast<cytnx_int64>();
           if (pybind_cytnx::is_string_args(args)) {
             return self.permute(pybind_cytnx::parse_index_args<std::string>(args), rowrank);
           }
           return self.permute(pybind_cytnx::parse_index_args<cytnx_int64>(args), rowrank);
         })

     .def("permute_nosignflip", [](UniTensor &self, const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank){
                        return self.permute_nosignflip(mapper,rowrank);
                },py::arg("mapper"), py::arg("rowrank")=(cytnx_int64)(-1))

    .def("permute_nosignflip", [](UniTensor &self, const std::vector<std::string> &mapper, const cytnx_int64 &rowrank){
                        return self.permute_nosignflip(mapper,rowrank);
                },py::arg("mapper"), py::arg("rowrank")=(cytnx_int64)(-1))
     .def("permute_nosignflip_", [](UniTensor &self, const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank){
                        return &self.permute_nosignflip_(mapper,rowrank);
                },py::arg("mapper"), py::arg("rowrank")=(cytnx_int64)(-1))

    .def("permute_nosignflip_", [](UniTensor &self, const std::vector<std::string> &mapper, const cytnx_int64 &rowrank){
                        return &self.permute_nosignflip_(mapper,rowrank);
                },py::arg("mapper"), py::arg("rowrank")=(cytnx_int64)(-1))

     .def("twist", [](UniTensor &self, const cytnx_int64 &idx){
                        return self.twist(idx);
                },py::arg("idx"))
     .def("twist", [](UniTensor &self, const std::string label){
                        return self.twist(label);
                },py::arg("label"))
     .def("twist_", [](UniTensor &self, const cytnx_int64 &idx){
                        return self.twist_(idx);
                },py::arg("idx"))
     .def("twist_", [](UniTensor &self, const std::string label){
                        return self.twist_(label);
                },py::arg("label"))
     .def("fermion_twists", &UniTensor::fermion_twists)
     .def("fermion_twists_", &UniTensor::fermion_twists_)

    .def("make_contiguous", &UniTensor::contiguous)
    .def("contiguous_", &UniTensor::contiguous_)
    .def("apply", &UniTensor::apply,
         "Apply fermionic signflips and return a new UniTensor. Blocks that require a signflip "
         "are copied and inverted; blocks that do not are shared views. Non-fermionic tensors are "
         "returned unchanged. See also: apply_()")
    .def("apply_", &UniTensor::apply_,
         "Apply fermionic signflips inplacely. Subsequently, signflip() returns False for all "
         "elements. Non-fermionic tensors are left unchanged. See also: apply()")
    .def("print_diagram", &UniTensor::print_diagram, py::arg("bond_info") = false,
         py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
    .def("print_blocks", &UniTensor::print_blocks, py::arg("full_info") = true,
         py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
    .def("print_block", &UniTensor::print_block, py::arg("idx"), py::arg("full_info") = true,
         py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())

    .def("group_basis_", &UniTensor::group_basis_)
    .def("group_basis", &UniTensor::group_basis)
    .def("get_block",
      [](const UniTensor &self, const cytnx_uint64 &idx) { return self.get_block(idx); },
      py::arg("idx") = (cytnx_uint64)(0))

    // numpy_scalar keep-set for the qnum parameter, ahead of the py::int_
    // dispatcher below -- see dispatch_numpy_int_vector's comment.
    .def(
      "get_block",
      [](const UniTensor &self, const std::vector<py::numpy_scalar<int64_t>> &qnum,
         const bool &force) {
        return dispatch_numpy_int_vector(qnum,
                                          [&](const auto &v) { return self.get_block(v, force); });
      },
      py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block",
      [](const UniTensor &self, const std::vector<std::string> &label,
         const std::vector<py::numpy_scalar<int64_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(
          qnum, [&](const auto &v) { return self.get_block(label, v, force); });
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block_",
      [](UniTensor &self, const std::vector<py::numpy_scalar<int64_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(qnum,
                                          [&](const auto &v) { return self.get_block_(v, force); });
      },
      py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block_",
      [](UniTensor &self, const std::vector<std::string> &labels,
         const std::vector<py::numpy_scalar<int64_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(
          qnum, [&](const auto &v) { return self.get_block_(labels, v, force); });
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block",
      [](const UniTensor &self, const std::vector<py::numpy_scalar<uint64_t>> &qnum,
         const bool &force) {
        return dispatch_numpy_int_vector(qnum,
                                          [&](const auto &v) { return self.get_block(v, force); });
      },
      py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block",
      [](const UniTensor &self, const std::vector<std::string> &label,
         const std::vector<py::numpy_scalar<uint64_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(
          qnum, [&](const auto &v) { return self.get_block(label, v, force); });
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block_",
      [](UniTensor &self, const std::vector<py::numpy_scalar<uint64_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(qnum,
                                          [&](const auto &v) { return self.get_block_(v, force); });
      },
      py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block_",
      [](UniTensor &self, const std::vector<std::string> &labels,
         const std::vector<py::numpy_scalar<uint64_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(
          qnum, [&](const auto &v) { return self.get_block_(labels, v, force); });
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block",
      [](const UniTensor &self, const std::vector<py::numpy_scalar<int32_t>> &qnum,
         const bool &force) {
        return dispatch_numpy_int_vector(qnum,
                                          [&](const auto &v) { return self.get_block(v, force); });
      },
      py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block",
      [](const UniTensor &self, const std::vector<std::string> &label,
         const std::vector<py::numpy_scalar<int32_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(
          qnum, [&](const auto &v) { return self.get_block(label, v, force); });
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block_",
      [](UniTensor &self, const std::vector<py::numpy_scalar<int32_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(qnum,
                                          [&](const auto &v) { return self.get_block_(v, force); });
      },
      py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block_",
      [](UniTensor &self, const std::vector<std::string> &labels,
         const std::vector<py::numpy_scalar<int32_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(
          qnum, [&](const auto &v) { return self.get_block_(labels, v, force); });
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block",
      [](const UniTensor &self, const std::vector<py::numpy_scalar<uint32_t>> &qnum,
         const bool &force) {
        return dispatch_numpy_int_vector(qnum,
                                          [&](const auto &v) { return self.get_block(v, force); });
      },
      py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block",
      [](const UniTensor &self, const std::vector<std::string> &label,
         const std::vector<py::numpy_scalar<uint32_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(
          qnum, [&](const auto &v) { return self.get_block(label, v, force); });
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block_",
      [](UniTensor &self, const std::vector<py::numpy_scalar<uint32_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(qnum,
                                          [&](const auto &v) { return self.get_block_(v, force); });
      },
      py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block_",
      [](UniTensor &self, const std::vector<std::string> &labels,
         const std::vector<py::numpy_scalar<uint32_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(
          qnum, [&](const auto &v) { return self.get_block_(labels, v, force); });
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block",
      [](const UniTensor &self, const std::vector<py::numpy_scalar<int16_t>> &qnum,
         const bool &force) {
        return dispatch_numpy_int_vector(qnum,
                                          [&](const auto &v) { return self.get_block(v, force); });
      },
      py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block",
      [](const UniTensor &self, const std::vector<std::string> &label,
         const std::vector<py::numpy_scalar<int16_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(
          qnum, [&](const auto &v) { return self.get_block(label, v, force); });
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block_",
      [](UniTensor &self, const std::vector<py::numpy_scalar<int16_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(qnum,
                                          [&](const auto &v) { return self.get_block_(v, force); });
      },
      py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block_",
      [](UniTensor &self, const std::vector<std::string> &labels,
         const std::vector<py::numpy_scalar<int16_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(
          qnum, [&](const auto &v) { return self.get_block_(labels, v, force); });
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block",
      [](const UniTensor &self, const std::vector<py::numpy_scalar<uint16_t>> &qnum,
         const bool &force) {
        return dispatch_numpy_int_vector(qnum,
                                          [&](const auto &v) { return self.get_block(v, force); });
      },
      py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block",
      [](const UniTensor &self, const std::vector<std::string> &label,
         const std::vector<py::numpy_scalar<uint16_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(
          qnum, [&](const auto &v) { return self.get_block(label, v, force); });
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block_",
      [](UniTensor &self, const std::vector<py::numpy_scalar<uint16_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(qnum,
                                          [&](const auto &v) { return self.get_block_(v, force); });
      },
      py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block_",
      [](UniTensor &self, const std::vector<std::string> &labels,
         const std::vector<py::numpy_scalar<uint16_t>> &qnum, const bool &force) {
        return dispatch_numpy_int_vector(
          qnum, [&](const auto &v) { return self.get_block_(labels, v, force); });
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)

    .def("get_block",
      [](const UniTensor &self, const std::vector<py::int_> &qnum, const bool &force) {
        return dispatch_pyint_vector(qnum, [&](const auto &v) { return self.get_block(v, force); });
      },
      py::arg("qnum"), py::arg("force") = false)

    .def("get_block",
      [](const UniTensor &self, const std::vector<std::string> &label,
         const std::vector<py::int_> &qnum, const bool &force) {
        return dispatch_pyint_vector(
          qnum, [&](const auto &v) { return self.get_block(label, v, force); });
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)
    .def("get_block_",
      [](UniTensor &self, const std::vector<py::int_> &qnum, const bool &force) {
        return dispatch_pyint_vector(qnum,
                                      [&](const auto &v) { return self.get_block_(v, force); });
      },
      py::arg("qnum"), py::arg("force") = false)

    .def("get_block_",
      [](UniTensor &self, const std::vector<std::string> &labels,
         const std::vector<py::int_> &qnum, const bool &force) {
        return dispatch_pyint_vector(
          qnum, [&](const auto &v) { return self.get_block_(labels, v, force); });
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)


    .def("get_block_", [](UniTensor &self, const cytnx_uint64 &idx) { return self.get_block_(idx); },
      py::arg("idx") = (cytnx_uint64)(0))
    .def("get_blocks", [](const UniTensor &self) { return self.get_blocks(); })
    .def("get_blocks_",
      [](UniTensor &self, py::args args, py::kwargs kwargs) {
        return self.get_blocks_(parse_get_blocks_silent_arg(args, kwargs));
      }
      // ,py::arg("silent") = false // Uncomment this line after removing the deprecated argument.
    )
    .def("put_block",
      [](UniTensor &self, const cytnx::Tensor &in, const cytnx_uint64 &idx) {
        self.put_block(in, idx);
      },
      py::arg("Tin"), py::arg("idx") = (cytnx_uint64)(0))

    .def("put_block",
      [](UniTensor &self, const cytnx::Tensor &in, const std::vector<cytnx_int64> &qnum) {
        self.put_block(in, qnum);
      },
      py::arg("Tin"), py::arg("qidx"))
    .def("put_block",
      [](UniTensor &self, cytnx::Tensor &in, const std::vector<std::string> &lbls, const std::vector<cytnx_int64> &qnum) {
        self.put_block(in, lbls, qnum);
      },
      py::arg("Tin"), py::arg("labels"), py::arg("qidx"))

    // [Deprecated force argument!]
    .def("put_block",
      [](UniTensor &self, const cytnx::Tensor &in, const std::vector<cytnx_int64> &qnum,
         const bool &force) {
          py::warnings::warn("Argument 'force' is deprecated and will be removed; use put_block(in, qnum) without force argument instead.",
                              PyExc_FutureWarning, 2);
          self.put_block(in, qnum, force);
      },
      py::arg("Tin"), py::arg("qidx"), py::arg("force"))
    // [Deprecated force argument!]
    .def("put_block",
      [](UniTensor &self, cytnx::Tensor &in, const std::vector<std::string> &lbls, const std::vector<cytnx_int64> &qnum,
         const bool &force) {
          py::warnings::warn("Argument 'force' is deprecated and will be removed; use put_block(Tin, labels, qidx) without force argument instead.",
                              PyExc_FutureWarning, 2);
          self.put_block(in, lbls, qnum, force);
      },
      py::arg("Tin"), py::arg("labels"), py::arg("qidx"), py::arg("force"))

    // [Deprecated 'in' keyword argument]
    // The block tensor used to be exposed under the Python keyword 'in'. Because
    // 'in' is a reserved word, put_block(in=...) was always a SyntaxError and the
    // argument could only be passed as put_block(**{"in": block}). Catch that form
    // through **kwargs so existing callers keep working, warn, and forward to the
    // 'Tin' parameter handled by the typed overloads above.
    .def("put_block",
      [](UniTensor &self, py::args args, py::kwargs kwargs) {
          if (!kwargs.contains("in"))
            throw py::type_error(
              "put_block(): incompatible arguments; see help(UniTensor.put_block).");
          if (kwargs.contains("Tin"))
            throw py::type_error(
              "put_block(): got both the deprecated 'in' and 'Tin' for the same "
              "argument; pass only 'Tin'.");
          py::warnings::warn(
            "The 'in' keyword argument of UniTensor.put_block is deprecated; use 'Tin' instead.",
            PyExc_FutureWarning, 2);
          py::dict forwarded;
          for (auto item : kwargs) {
            if (py::cast<std::string>(item.first) == "in")
              forwarded["Tin"] = item.second;
            else
              forwarded[item.first] = item.second;
          }
          py::object pyself = py::cast(self, py::return_value_policy::reference);
          return pyself.attr("put_block")(*args, **forwarded);
      })

    .def("put_block_",
      [](UniTensor &self, cytnx::Tensor &in, const cytnx_uint64 &idx) { self.put_block_(in, idx); },
      py::arg("Tin"), py::arg("idx") = (cytnx_uint64)(0))

    .def("put_block_",
      [](UniTensor &self, cytnx::Tensor &in, const std::vector<cytnx_int64> &qnum) {
        self.put_block_(in, qnum);
      },
      py::arg("Tin"), py::arg("qidx"))
    .def("put_block_",
      [](UniTensor &self, cytnx::Tensor &in, const std::vector<std::string> &lbls, const std::vector<cytnx_int64> &qnum) {
        self.put_block_(in, lbls, qnum);
      },
      py::arg("Tin"), py::arg("labels"), py::arg("qidx"))

    // [Deprecated force argument!]
    .def("put_block_",
      [](UniTensor &self, cytnx::Tensor &in, const std::vector<cytnx_int64> &qnum,
         const bool &force) {
          py::warnings::warn("Argument 'force' is deprecated and will be removed; use put_block_(Tin, qidx) without force argument instead.",
                              PyExc_FutureWarning, 2);
          self.put_block_(in, qnum, force);
      },
      py::arg("Tin"), py::arg("qidx"), py::arg("force"))
    // [Deprecated force argument!]
    .def("put_block_",
      [](UniTensor &self, cytnx::Tensor &in, const std::vector<std::string> &lbls, const std::vector<cytnx_int64> &qnum,
         const bool &force) {
          py::warnings::warn("Argument 'force' is deprecated and will be removed; use put_block_(Tin, labels, qidx) without force argument instead.",
                              PyExc_FutureWarning, 2);
          self.put_block_(in, lbls, qnum, force);
      },
      py::arg("Tin"), py::arg("labels"), py::arg("qidx"), py::arg("force"))

    // [Deprecated 'in' keyword argument] See the put_block note above; accept the
    // legacy put_block_(**{"in": block}) form, warn, and forward to 'Tin'.
    .def("put_block_",
      [](UniTensor &self, py::args args, py::kwargs kwargs) {
          if (!kwargs.contains("in"))
            throw py::type_error(
              "put_block_(): incompatible arguments; see help(UniTensor.put_block_).");
          if (kwargs.contains("Tin"))
            throw py::type_error(
              "put_block_(): got both the deprecated 'in' and 'Tin' for the same "
              "argument; pass only 'Tin'.");
          py::warnings::warn(
            "The 'in' keyword argument of UniTensor.put_block_ is deprecated; use 'Tin' instead.",
            PyExc_FutureWarning, 2);
          py::dict forwarded;
          for (auto item : kwargs) {
            if (py::cast<std::string>(item.first) == "in")
              forwarded["Tin"] = item.second;
            else
              forwarded[item.first] = item.second;
          }
          py::object pyself = py::cast(self, py::return_value_policy::reference);
          return pyself.attr("put_block_")(*args, **forwarded);
      })

    .def("__repr__",
      [](UniTensor &self) -> std::string {
        std::cout << self << std::endl;
        return std::string("");
      },
      py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
    .def("to_dense", &UniTensor::to_dense)
    .def("to_dense_", &UniTensor::to_dense_)
    // combineBonds() is deprecated (#421/#422): use combineBond_() (in-place) or combineBond()
    // (out-of-place) instead. Deprecation-warning suppression lives in the file-scope helpers
    // (a #pragma GCC diagnostic cannot legally sit inside this .def() chain expression under GCC).
    .def("combineBonds", &unitensor_combineBonds_int_deprecated, py::arg("indicators"),
         py::arg("force") = false, py::arg("by_label") = false)
    .def("combineBonds", &unitensor_combineBonds_str_deprecated, py::arg("indicators"),
         py::arg("force") = false)

    .def("combineBond_",
         [](py::object self, const std::vector<std::string> &indicators, const bool &force)
         {
            self.cast<UniTensor &>().combineBond_(indicators,force);
            return self;
         },
         py::arg("indicators"), py::arg("force") = false)

    .def("combineBond",
         [](const UniTensor &self, const std::vector<std::string> &indicators, const bool &force)
         {
            return self.combineBond(indicators,force);
         },
         py::arg("indicators"), py::arg("force") = false)



    // GIL: see the guard discipline note in linalg_py.cpp
    .def("contract", &UniTensor::contract, py::arg("inR"), py::arg("mv_elem_self")=false, py::arg("mv_elem_rhs")=false, py::call_guard<py::gil_scoped_release>())

    .def("getTotalQnums", &UniTensor::getTotalQnums, py::arg("physical")=false)

    .def("get_blocks_qnums", &UniTensor::get_blocks_qnums)

    // arithmetic >>
    .def("__neg__",
         [](UniTensor &self) {
           if (self.dtype() == Type.Double) {
             return linalg::Mul(cytnx_double(-1), self);
           } else if (self.dtype() == Type.ComplexDouble) {
             return linalg::Mul(cytnx_complex128(-1, 0), self);
           } else if (self.dtype() == Type.Float) {
             return linalg::Mul(cytnx_float(-1), self);
           } else if (self.dtype() == Type.ComplexFloat) {
             return linalg::Mul(cytnx_complex64(-1, 0), self);
           } else {
             return linalg::Mul(-1, self);
           }
         })
    .def("__pos__", [](UniTensor &self) { return self; })
    // NOTE: no __eq__/__ne__/__bool__ on UniTensor (intentional, Task 3 scope) -- Python falls
    // back to identity semantics; see Tensor's __ne__/__bool__ notes in tensor_py.cpp.
    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    // Phase-2 Task 3 (#934/2026-07-06 decision, amended): UniTensor+UniTensor is kept but
    // guarded on matching metadata; see unitensor_addsub_metadata_mismatch's doc comment above.
    .def("__add__",
         [](UniTensor &self, const UniTensor &rhs) -> UniTensor {
           const std::string reason = unitensor_addsub_metadata_mismatch(self, rhs);
           if (!reason.empty()) raise_unitensor_addsub_metadata_mismatch(" + ", reason);
           // Add() dtype-promotes but resets labels to a plain range and clears the name;
           // restore the shared metadata since the operands matched.
           UniTensor out = self.Add(rhs);
           out.relabel_(self.labels());
           out.set_name_(self.name());
           return out;
         })
    .def("__add__",
         [](UniTensor &self, const py::numpy_scalar<float> &rhs) {
           return linalg::Add(self, static_cast<cytnx::cytnx_float>(rhs));
         })
    .def("__add__",
         [](UniTensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return linalg::Add(self, static_cast<cytnx::cytnx_complex64>(rhs));
         })
    .def("__add__",
         [](UniTensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return linalg::Add(self, static_cast<cytnx::cytnx_int64>(rhs));
         })
    .def("__add__",
         [](UniTensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return linalg::Add(self, static_cast<cytnx::cytnx_uint64>(rhs));
         })
    .def("__add__",
         [](UniTensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return linalg::Add(self, static_cast<cytnx::cytnx_int32>(rhs));
         })
    .def("__add__",
         [](UniTensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return linalg::Add(self, static_cast<cytnx::cytnx_uint32>(rhs));
         })
    .def("__add__",
         [](UniTensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return linalg::Add(self, static_cast<cytnx::cytnx_int16>(rhs));
         })
    .def("__add__",
         [](UniTensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return linalg::Add(self, static_cast<cytnx::cytnx_uint16>(rhs));
         })
    .def("__add__",
         [](UniTensor &self, const py::numpy_scalar<bool> &rhs) {
           return linalg::Add(self, static_cast<cytnx::cytnx_bool>(rhs));
         })
    .def("__add__",
         [](UniTensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) { return linalg::Add(self, v); });
         })
    .def("__add__",
         [](UniTensor &self, const cytnx::cytnx_double &rhs) { return linalg::Add(self, rhs); })
    .def("__add__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return linalg::Add(self, rhs); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("__radd__",
         [](UniTensor &self, const py::numpy_scalar<float> &lhs) {
           return linalg::Add(static_cast<cytnx::cytnx_float>(lhs), self);
         })
    .def("__radd__",
         [](UniTensor &self, const py::numpy_scalar<std::complex<float>> &lhs) {
           return linalg::Add(static_cast<cytnx::cytnx_complex64>(lhs), self);
         })
    .def("__radd__",
         [](UniTensor &self, const py::numpy_scalar<int64_t> &lhs) {
           return linalg::Add(static_cast<cytnx::cytnx_int64>(lhs), self);
         })
    .def("__radd__",
         [](UniTensor &self, const py::numpy_scalar<uint64_t> &lhs) {
           return linalg::Add(static_cast<cytnx::cytnx_uint64>(lhs), self);
         })
    .def("__radd__",
         [](UniTensor &self, const py::numpy_scalar<int32_t> &lhs) {
           return linalg::Add(static_cast<cytnx::cytnx_int32>(lhs), self);
         })
    .def("__radd__",
         [](UniTensor &self, const py::numpy_scalar<uint32_t> &lhs) {
           return linalg::Add(static_cast<cytnx::cytnx_uint32>(lhs), self);
         })
    .def("__radd__",
         [](UniTensor &self, const py::numpy_scalar<int16_t> &lhs) {
           return linalg::Add(static_cast<cytnx::cytnx_int16>(lhs), self);
         })
    .def("__radd__",
         [](UniTensor &self, const py::numpy_scalar<uint16_t> &lhs) {
           return linalg::Add(static_cast<cytnx::cytnx_uint16>(lhs), self);
         })
    .def("__radd__",
         [](UniTensor &self, const py::numpy_scalar<bool> &lhs) {
           return linalg::Add(static_cast<cytnx::cytnx_bool>(lhs), self);
         })
    .def("__radd__",
         [](UniTensor &self, const py::int_ &lhs) {
           return dispatch_pyint(lhs, [&](auto v) { return linalg::Add(v, self); });
         })
    .def("__radd__",
         [](UniTensor &self, const cytnx::cytnx_double &lhs) { return linalg::Add(lhs, self); })
    .def("__radd__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &lhs) { return linalg::Add(lhs, self); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    // Phase-2 Task 3 (#934/2026-07-06 decision, amended): UniTensor+=UniTensor is kept but
    // guarded on matching metadata; see unitensor_addsub_metadata_mismatch's doc comment above.
    .def("__iadd__",
         [](UniTensor &self, const UniTensor &rhs) -> UniTensor & {
           const std::string reason = unitensor_addsub_metadata_mismatch(self, rhs);
           if (!reason.empty()) raise_unitensor_addsub_metadata_mismatch(" += ", reason);
           return self.Add_(rhs);  // in-place; preserves self's labels and name
         })
    .def("__iadd__",
         [](UniTensor &self, const py::numpy_scalar<float> &rhs) {
           return self.Add_(static_cast<cytnx::cytnx_float>(rhs));
         })
    .def("__iadd__",
         [](UniTensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return self.Add_(static_cast<cytnx::cytnx_complex64>(rhs));
         })
    .def("__iadd__",
         [](UniTensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return self.Add_(static_cast<cytnx::cytnx_int64>(rhs));
         })
    .def("__iadd__",
         [](UniTensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return self.Add_(static_cast<cytnx::cytnx_uint64>(rhs));
         })
    .def("__iadd__",
         [](UniTensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return self.Add_(static_cast<cytnx::cytnx_int32>(rhs));
         })
    .def("__iadd__",
         [](UniTensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return self.Add_(static_cast<cytnx::cytnx_uint32>(rhs));
         })
    .def("__iadd__",
         [](UniTensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return self.Add_(static_cast<cytnx::cytnx_int16>(rhs));
         })
    .def("__iadd__",
         [](UniTensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return self.Add_(static_cast<cytnx::cytnx_uint16>(rhs));
         })
    .def("__iadd__",
         [](UniTensor &self, const py::numpy_scalar<bool> &rhs) {
           return self.Add_(static_cast<cytnx::cytnx_bool>(rhs));
         })
    .def("__iadd__",
         [](UniTensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) { return self.Add_(v); });
         })
    .def("__iadd__",
         [](UniTensor &self, const cytnx::cytnx_double &rhs) { return self.Add_(rhs); })
    .def("__iadd__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Add_(rhs); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    // Phase-2 Task 3 (#934/2026-07-06 decision, amended): UniTensor-UniTensor is kept but
    // guarded on matching metadata; see unitensor_addsub_metadata_mismatch's doc comment above.
    .def("__sub__",
         [](UniTensor &self, const UniTensor &rhs) -> UniTensor {
           const std::string reason = unitensor_addsub_metadata_mismatch(self, rhs);
           if (!reason.empty()) raise_unitensor_addsub_metadata_mismatch(" - ", reason);
           // Sub() dtype-promotes but resets labels to a plain range and clears the name;
           // restore the shared metadata since the operands matched.
           UniTensor out = self.Sub(rhs);
           out.relabel_(self.labels());
           out.set_name_(self.name());
           return out;
         })
    .def("__sub__",
         [](UniTensor &self, const py::numpy_scalar<float> &rhs) {
           return linalg::Sub(self, static_cast<cytnx::cytnx_float>(rhs));
         })
    .def("__sub__",
         [](UniTensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return linalg::Sub(self, static_cast<cytnx::cytnx_complex64>(rhs));
         })
    .def("__sub__",
         [](UniTensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return linalg::Sub(self, static_cast<cytnx::cytnx_int64>(rhs));
         })
    .def("__sub__",
         [](UniTensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return linalg::Sub(self, static_cast<cytnx::cytnx_uint64>(rhs));
         })
    .def("__sub__",
         [](UniTensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return linalg::Sub(self, static_cast<cytnx::cytnx_int32>(rhs));
         })
    .def("__sub__",
         [](UniTensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return linalg::Sub(self, static_cast<cytnx::cytnx_uint32>(rhs));
         })
    .def("__sub__",
         [](UniTensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return linalg::Sub(self, static_cast<cytnx::cytnx_int16>(rhs));
         })
    .def("__sub__",
         [](UniTensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return linalg::Sub(self, static_cast<cytnx::cytnx_uint16>(rhs));
         })
    .def("__sub__",
         [](UniTensor &self, const py::numpy_scalar<bool> &rhs) {
           return linalg::Sub(self, static_cast<cytnx::cytnx_bool>(rhs));
         })
    .def("__sub__",
         [](UniTensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) { return linalg::Sub(self, v); });
         })
    .def("__sub__",
         [](UniTensor &self, const cytnx::cytnx_double &rhs) { return linalg::Sub(self, rhs); })
    .def("__sub__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return linalg::Sub(self, rhs); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("__rsub__",
         [](UniTensor &self, const py::numpy_scalar<float> &lhs) {
           return linalg::Sub(static_cast<cytnx::cytnx_float>(lhs), self);
         })
    .def("__rsub__",
         [](UniTensor &self, const py::numpy_scalar<std::complex<float>> &lhs) {
           return linalg::Sub(static_cast<cytnx::cytnx_complex64>(lhs), self);
         })
    .def("__rsub__",
         [](UniTensor &self, const py::numpy_scalar<int64_t> &lhs) {
           return linalg::Sub(static_cast<cytnx::cytnx_int64>(lhs), self);
         })
    .def("__rsub__",
         [](UniTensor &self, const py::numpy_scalar<uint64_t> &lhs) {
           return linalg::Sub(static_cast<cytnx::cytnx_uint64>(lhs), self);
         })
    .def("__rsub__",
         [](UniTensor &self, const py::numpy_scalar<int32_t> &lhs) {
           return linalg::Sub(static_cast<cytnx::cytnx_int32>(lhs), self);
         })
    .def("__rsub__",
         [](UniTensor &self, const py::numpy_scalar<uint32_t> &lhs) {
           return linalg::Sub(static_cast<cytnx::cytnx_uint32>(lhs), self);
         })
    .def("__rsub__",
         [](UniTensor &self, const py::numpy_scalar<int16_t> &lhs) {
           return linalg::Sub(static_cast<cytnx::cytnx_int16>(lhs), self);
         })
    .def("__rsub__",
         [](UniTensor &self, const py::numpy_scalar<uint16_t> &lhs) {
           return linalg::Sub(static_cast<cytnx::cytnx_uint16>(lhs), self);
         })
    .def("__rsub__",
         [](UniTensor &self, const py::numpy_scalar<bool> &lhs) {
           return linalg::Sub(static_cast<cytnx::cytnx_bool>(lhs), self);
         })
    .def("__rsub__",
         [](UniTensor &self, const py::int_ &lhs) {
           return dispatch_pyint(lhs, [&](auto v) { return linalg::Sub(v, self); });
         })
    .def("__rsub__",
         [](UniTensor &self, const cytnx::cytnx_double &lhs) { return linalg::Sub(lhs, self); })
    .def("__rsub__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &lhs) { return linalg::Sub(lhs, self); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    // Phase-2 Task 3 (#934/2026-07-06 decision, amended): UniTensor-=UniTensor is kept but
    // guarded on matching metadata; see unitensor_addsub_metadata_mismatch's doc comment above.
    .def("__isub__",
         [](UniTensor &self, const UniTensor &rhs) -> UniTensor & {
           const std::string reason = unitensor_addsub_metadata_mismatch(self, rhs);
           if (!reason.empty()) raise_unitensor_addsub_metadata_mismatch(" -= ", reason);
           return self.Sub_(rhs);  // in-place; preserves self's labels and name
         })
    .def("__isub__",
         [](UniTensor &self, const py::numpy_scalar<float> &rhs) {
           return self.Sub_(static_cast<cytnx::cytnx_float>(rhs));
         })
    .def("__isub__",
         [](UniTensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return self.Sub_(static_cast<cytnx::cytnx_complex64>(rhs));
         })
    .def("__isub__",
         [](UniTensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return self.Sub_(static_cast<cytnx::cytnx_int64>(rhs));
         })
    .def("__isub__",
         [](UniTensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return self.Sub_(static_cast<cytnx::cytnx_uint64>(rhs));
         })
    .def("__isub__",
         [](UniTensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return self.Sub_(static_cast<cytnx::cytnx_int32>(rhs));
         })
    .def("__isub__",
         [](UniTensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return self.Sub_(static_cast<cytnx::cytnx_uint32>(rhs));
         })
    .def("__isub__",
         [](UniTensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return self.Sub_(static_cast<cytnx::cytnx_int16>(rhs));
         })
    .def("__isub__",
         [](UniTensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return self.Sub_(static_cast<cytnx::cytnx_uint16>(rhs));
         })
    .def("__isub__",
         [](UniTensor &self, const py::numpy_scalar<bool> &rhs) {
           return self.Sub_(static_cast<cytnx::cytnx_bool>(rhs));
         })
    .def("__isub__",
         [](UniTensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) { return self.Sub_(v); });
         })
    .def("__isub__",
         [](UniTensor &self, const cytnx::cytnx_double &rhs) { return self.Sub_(rhs); })
    .def("__isub__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Sub_(rhs); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    // Phase-2 Task 3 (#934/2026-07-06 decision): UniTensor*UniTensor (Hadamard/elementwise
    // product) removed from the python surface -- it is basis-dependent and not a
    // tensor-network operation (#934); see raise_unitensor_elementwise_removed above.
    .def("__mul__",
         [](UniTensor &self, const UniTensor &rhs) -> UniTensor {
           raise_unitensor_elementwise_removed(" * ", kUniTensorMulRemovedGuidance);
         })
    .def("__mul__",
         [](UniTensor &self, const py::numpy_scalar<float> &rhs) {
           return linalg::Mul(self, static_cast<cytnx::cytnx_float>(rhs));
         })
    .def("__mul__",
         [](UniTensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return linalg::Mul(self, static_cast<cytnx::cytnx_complex64>(rhs));
         })
    .def("__mul__",
         [](UniTensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return linalg::Mul(self, static_cast<cytnx::cytnx_int64>(rhs));
         })
    .def("__mul__",
         [](UniTensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return linalg::Mul(self, static_cast<cytnx::cytnx_uint64>(rhs));
         })
    .def("__mul__",
         [](UniTensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return linalg::Mul(self, static_cast<cytnx::cytnx_int32>(rhs));
         })
    .def("__mul__",
         [](UniTensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return linalg::Mul(self, static_cast<cytnx::cytnx_uint32>(rhs));
         })
    .def("__mul__",
         [](UniTensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return linalg::Mul(self, static_cast<cytnx::cytnx_int16>(rhs));
         })
    .def("__mul__",
         [](UniTensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return linalg::Mul(self, static_cast<cytnx::cytnx_uint16>(rhs));
         })
    .def("__mul__",
         [](UniTensor &self, const py::numpy_scalar<bool> &rhs) {
           return linalg::Mul(self, static_cast<cytnx::cytnx_bool>(rhs));
         })
    .def("__mul__",
         [](UniTensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) { return linalg::Mul(self, v); });
         })
    .def("__mul__",
         [](UniTensor &self, const cytnx::cytnx_double &rhs) { return linalg::Mul(self, rhs); })
    .def("__mul__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return linalg::Mul(self, rhs); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("__rmul__",
         [](UniTensor &self, const py::numpy_scalar<float> &lhs) {
           return linalg::Mul(static_cast<cytnx::cytnx_float>(lhs), self);
         })
    .def("__rmul__",
         [](UniTensor &self, const py::numpy_scalar<std::complex<float>> &lhs) {
           return linalg::Mul(static_cast<cytnx::cytnx_complex64>(lhs), self);
         })
    .def("__rmul__",
         [](UniTensor &self, const py::numpy_scalar<int64_t> &lhs) {
           return linalg::Mul(static_cast<cytnx::cytnx_int64>(lhs), self);
         })
    .def("__rmul__",
         [](UniTensor &self, const py::numpy_scalar<uint64_t> &lhs) {
           return linalg::Mul(static_cast<cytnx::cytnx_uint64>(lhs), self);
         })
    .def("__rmul__",
         [](UniTensor &self, const py::numpy_scalar<int32_t> &lhs) {
           return linalg::Mul(static_cast<cytnx::cytnx_int32>(lhs), self);
         })
    .def("__rmul__",
         [](UniTensor &self, const py::numpy_scalar<uint32_t> &lhs) {
           return linalg::Mul(static_cast<cytnx::cytnx_uint32>(lhs), self);
         })
    .def("__rmul__",
         [](UniTensor &self, const py::numpy_scalar<int16_t> &lhs) {
           return linalg::Mul(static_cast<cytnx::cytnx_int16>(lhs), self);
         })
    .def("__rmul__",
         [](UniTensor &self, const py::numpy_scalar<uint16_t> &lhs) {
           return linalg::Mul(static_cast<cytnx::cytnx_uint16>(lhs), self);
         })
    .def("__rmul__",
         [](UniTensor &self, const py::numpy_scalar<bool> &lhs) {
           return linalg::Mul(static_cast<cytnx::cytnx_bool>(lhs), self);
         })
    .def("__rmul__",
         [](UniTensor &self, const py::int_ &lhs) {
           return dispatch_pyint(lhs, [&](auto v) { return linalg::Mul(v, self); });
         })
    .def("__rmul__",
         [](UniTensor &self, const cytnx::cytnx_double &lhs) { return linalg::Mul(lhs, self); })
    .def("__rmul__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &lhs) { return linalg::Mul(lhs, self); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    // Phase-2 Task 3 (#934/2026-07-06 decision): UniTensor*=UniTensor (Hadamard/elementwise
    // product) removed from the python surface; see __mul__'s comment above.
    .def("__imul__",
         [](UniTensor &self, const UniTensor &rhs) -> UniTensor & {
           raise_unitensor_elementwise_removed(" *= ", kUniTensorMulRemovedGuidance);
         })
    .def("__imul__",
         [](UniTensor &self, const py::numpy_scalar<float> &rhs) {
           return self.Mul_(static_cast<cytnx::cytnx_float>(rhs));
         })
    .def("__imul__",
         [](UniTensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return self.Mul_(static_cast<cytnx::cytnx_complex64>(rhs));
         })
    .def("__imul__",
         [](UniTensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return self.Mul_(static_cast<cytnx::cytnx_int64>(rhs));
         })
    .def("__imul__",
         [](UniTensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return self.Mul_(static_cast<cytnx::cytnx_uint64>(rhs));
         })
    .def("__imul__",
         [](UniTensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return self.Mul_(static_cast<cytnx::cytnx_int32>(rhs));
         })
    .def("__imul__",
         [](UniTensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return self.Mul_(static_cast<cytnx::cytnx_uint32>(rhs));
         })
    .def("__imul__",
         [](UniTensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return self.Mul_(static_cast<cytnx::cytnx_int16>(rhs));
         })
    .def("__imul__",
         [](UniTensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return self.Mul_(static_cast<cytnx::cytnx_uint16>(rhs));
         })
    .def("__imul__",
         [](UniTensor &self, const py::numpy_scalar<bool> &rhs) {
           return self.Mul_(static_cast<cytnx::cytnx_bool>(rhs));
         })
    .def("__imul__",
         [](UniTensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) { return self.Mul_(v); });
         })
    .def("__imul__",
         [](UniTensor &self, const cytnx::cytnx_double &rhs) { return self.Mul_(rhs); })
    .def("__imul__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Mul_(rhs); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    // Phase-2 Task 3 (#934/2026-07-06 decision): UniTensor/UniTensor (elementwise division)
    // removed from the python surface -- per #934 it has no well-defined tensor-network
    // meaning (basis-dependent, and typically produces inf/nan); see
    // raise_unitensor_elementwise_removed's doc comment above.
    .def("__truediv__",
         [](UniTensor &self, const UniTensor &rhs) -> UniTensor {
           raise_unitensor_elementwise_removed(" / ", kUniTensorDivRemovedGuidance);
         })
    .def("__truediv__",
         [](UniTensor &self, const py::numpy_scalar<float> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_float>(rhs));
         })
    .def("__truediv__",
         [](UniTensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_complex64>(rhs));
         })
    .def("__truediv__",
         [](UniTensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_int64>(rhs));
         })
    .def("__truediv__",
         [](UniTensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_uint64>(rhs));
         })
    .def("__truediv__",
         [](UniTensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_int32>(rhs));
         })
    .def("__truediv__",
         [](UniTensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_uint32>(rhs));
         })
    .def("__truediv__",
         [](UniTensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_int16>(rhs));
         })
    .def("__truediv__",
         [](UniTensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_uint16>(rhs));
         })
    .def("__truediv__",
         [](UniTensor &self, const py::numpy_scalar<bool> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_bool>(rhs));
         })
    .def("__truediv__",
         [](UniTensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) { return linalg::Div(self, v); });
         })
    .def("__truediv__",
         [](UniTensor &self, const cytnx::cytnx_double &rhs) { return linalg::Div(self, rhs); })
    .def("__truediv__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return linalg::Div(self, rhs); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("__rtruediv__",
         [](UniTensor &self, const py::numpy_scalar<float> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_float>(lhs), self);
         })
    .def("__rtruediv__",
         [](UniTensor &self, const py::numpy_scalar<std::complex<float>> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_complex64>(lhs), self);
         })
    .def("__rtruediv__",
         [](UniTensor &self, const py::numpy_scalar<int64_t> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_int64>(lhs), self);
         })
    .def("__rtruediv__",
         [](UniTensor &self, const py::numpy_scalar<uint64_t> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_uint64>(lhs), self);
         })
    .def("__rtruediv__",
         [](UniTensor &self, const py::numpy_scalar<int32_t> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_int32>(lhs), self);
         })
    .def("__rtruediv__",
         [](UniTensor &self, const py::numpy_scalar<uint32_t> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_uint32>(lhs), self);
         })
    .def("__rtruediv__",
         [](UniTensor &self, const py::numpy_scalar<int16_t> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_int16>(lhs), self);
         })
    .def("__rtruediv__",
         [](UniTensor &self, const py::numpy_scalar<uint16_t> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_uint16>(lhs), self);
         })
    .def("__rtruediv__",
         [](UniTensor &self, const py::numpy_scalar<bool> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_bool>(lhs), self);
         })
    .def("__rtruediv__",
         [](UniTensor &self, const py::int_ &lhs) {
           return dispatch_pyint(lhs, [&](auto v) { return linalg::Div(v, self); });
         })
    .def("__rtruediv__",
         [](UniTensor &self, const cytnx::cytnx_double &lhs) { return linalg::Div(lhs, self); })
    .def("__rtruediv__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &lhs) { return linalg::Div(lhs, self); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    // Phase-2 Task 3 (#934/2026-07-06 decision): UniTensor/=UniTensor (elementwise
    // division) removed from the python surface; see __truediv__'s comment above.
    .def("__itruediv__",
         [](UniTensor &self, const UniTensor &rhs) -> UniTensor & {
           raise_unitensor_elementwise_removed(" /= ", kUniTensorDivRemovedGuidance);
         })
    .def("__itruediv__",
         [](UniTensor &self, const py::numpy_scalar<float> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_float>(rhs));
         })
    .def("__itruediv__",
         [](UniTensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_complex64>(rhs));
         })
    .def("__itruediv__",
         [](UniTensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_int64>(rhs));
         })
    .def("__itruediv__",
         [](UniTensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_uint64>(rhs));
         })
    .def("__itruediv__",
         [](UniTensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_int32>(rhs));
         })
    .def("__itruediv__",
         [](UniTensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_uint32>(rhs));
         })
    .def("__itruediv__",
         [](UniTensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_int16>(rhs));
         })
    .def("__itruediv__",
         [](UniTensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_uint16>(rhs));
         })
    .def("__itruediv__",
         [](UniTensor &self, const py::numpy_scalar<bool> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_bool>(rhs));
         })
    .def("__itruediv__",
         [](UniTensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) { return self.Div_(v); });
         })
    .def("__itruediv__",
         [](UniTensor &self, const cytnx::cytnx_double &rhs) { return self.Div_(rhs); })
    .def("__itruediv__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Div_(rhs); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    // Python '//' maps to __floordiv__, a distinct dunder from '/'. It must mirror __truediv__:
    // UniTensor//UniTensor is the same removed elementwise (Hadamard) division (#934), so it raises
    // rather than silently routing to linalg::Div -- otherwise 'a // b' is a back-door for the
    // quotient that 'a / b' rejects. Scalar floordiv ('ut // 2.0') stays, matching scalar '/'.
    .def("__floordiv__",
         [](UniTensor &self, const UniTensor &rhs) -> UniTensor {
           raise_unitensor_elementwise_removed(" // ", kUniTensorDivRemovedGuidance);
         })
    .def("__floordiv__",
         [](UniTensor &self, const py::numpy_scalar<float> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_float>(rhs));
         })
    .def("__floordiv__",
         [](UniTensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_complex64>(rhs));
         })
    .def("__floordiv__",
         [](UniTensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_int64>(rhs));
         })
    .def("__floordiv__",
         [](UniTensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_uint64>(rhs));
         })
    .def("__floordiv__",
         [](UniTensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_int32>(rhs));
         })
    .def("__floordiv__",
         [](UniTensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_uint32>(rhs));
         })
    .def("__floordiv__",
         [](UniTensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_int16>(rhs));
         })
    .def("__floordiv__",
         [](UniTensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_uint16>(rhs));
         })
    .def("__floordiv__",
         [](UniTensor &self, const py::numpy_scalar<bool> &rhs) {
           return linalg::Div(self, static_cast<cytnx::cytnx_bool>(rhs));
         })
    .def("__floordiv__",
         [](UniTensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) { return linalg::Div(self, v); });
         })
    .def("__floordiv__",
         [](UniTensor &self, const cytnx::cytnx_double &rhs) { return linalg::Div(self, rhs); })
    .def("__floordiv__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return linalg::Div(self, rhs); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("__rfloordiv__",
         [](UniTensor &self, const py::numpy_scalar<float> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_float>(lhs), self);
         })
    .def("__rfloordiv__",
         [](UniTensor &self, const py::numpy_scalar<std::complex<float>> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_complex64>(lhs), self);
         })
    .def("__rfloordiv__",
         [](UniTensor &self, const py::numpy_scalar<int64_t> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_int64>(lhs), self);
         })
    .def("__rfloordiv__",
         [](UniTensor &self, const py::numpy_scalar<uint64_t> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_uint64>(lhs), self);
         })
    .def("__rfloordiv__",
         [](UniTensor &self, const py::numpy_scalar<int32_t> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_int32>(lhs), self);
         })
    .def("__rfloordiv__",
         [](UniTensor &self, const py::numpy_scalar<uint32_t> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_uint32>(lhs), self);
         })
    .def("__rfloordiv__",
         [](UniTensor &self, const py::numpy_scalar<int16_t> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_int16>(lhs), self);
         })
    .def("__rfloordiv__",
         [](UniTensor &self, const py::numpy_scalar<uint16_t> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_uint16>(lhs), self);
         })
    .def("__rfloordiv__",
         [](UniTensor &self, const py::numpy_scalar<bool> &lhs) {
           return linalg::Div(static_cast<cytnx::cytnx_bool>(lhs), self);
         })
    .def("__rfloordiv__",
         [](UniTensor &self, const py::int_ &lhs) {
           return dispatch_pyint(lhs, [&](auto v) { return linalg::Div(v, self); });
         })
    .def("__rfloordiv__",
         [](UniTensor &self, const cytnx::cytnx_double &lhs) { return linalg::Div(lhs, self); })
    .def("__rfloordiv__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &lhs) { return linalg::Div(lhs, self); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    // '//=' maps to __ifloordiv__; mirror __itruediv__ -- UniTensor//=UniTensor is the removed
    // in-place elementwise division (#934), so it raises instead of routing to Div_.
    .def("__ifloordiv__",
         [](UniTensor &self, const UniTensor &rhs) -> UniTensor & {
           raise_unitensor_elementwise_removed(" //= ", kUniTensorDivRemovedGuidance);
         })
    .def("__ifloordiv__",
         [](UniTensor &self, const py::numpy_scalar<float> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_float>(rhs));
         })
    .def("__ifloordiv__",
         [](UniTensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_complex64>(rhs));
         })
    .def("__ifloordiv__",
         [](UniTensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_int64>(rhs));
         })
    .def("__ifloordiv__",
         [](UniTensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_uint64>(rhs));
         })
    .def("__ifloordiv__",
         [](UniTensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_int32>(rhs));
         })
    .def("__ifloordiv__",
         [](UniTensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_uint32>(rhs));
         })
    .def("__ifloordiv__",
         [](UniTensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_int16>(rhs));
         })
    .def("__ifloordiv__",
         [](UniTensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_uint16>(rhs));
         })
    .def("__ifloordiv__",
         [](UniTensor &self, const py::numpy_scalar<bool> &rhs) {
           return self.Div_(static_cast<cytnx::cytnx_bool>(rhs));
         })
    .def("__ifloordiv__",
         [](UniTensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) { return self.Div_(v); });
         })
    .def("__ifloordiv__",
         [](UniTensor &self, const cytnx::cytnx_double &rhs) { return self.Div_(rhs); })
    .def("__ifloordiv__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Div_(rhs); })


    /*
    .def("__mod__",[](UniTensor &self, const UniTensor &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_complex128&rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_double    &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_float     &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_int64     &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_int32     &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_int16     &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_bool    &rhs){return self.Mod(rhs);})
    .def("__mod__", [](cytnx::Tensor &self, const cytnx::Scalar &lhs) { return self.Mod(lhs, self); })
    .def("__mod__", [](cytnx::Tensor &self, const py::numpy_scalar<std::complex<double>> &lhs) { return self.Mod(static_cast<cytnx::cytnx_complex128>(lhs), self); })
    .def("__mod__", [](cytnx::Tensor &self, const py::numpy_scalar<std::complex<float>> &lhs) { return self.Mod(static_cast<cytnx::cytnx_complex64>(lhs), self); })
    .def("__mod__", [](cytnx::Tensor &self, const py::numpy_scalar<double> &lhs) { return self.Mod(static_cast<cytnx::cytnx_double>(lhs), self); })
    .def("__mod__", [](cytnx::UniTensor &self, const py::numpy_scalar<float> &lhs) { return self.Mod(static_cast<cytnx::cytnx_float>(lhs), self); })
    .def("__mod__", [](cytnx::UniTensor &self, const py::numpy_scalar<int64_t> &lhs) { return self.Mod(static_cast<cytnx::cytnx_int64>(lhs), self); })
    .def("__mod__", [](cytnx::UniTensor &self, const py::numpy_scalar<uint64_t> &lhs) { return self.Mod(static_cast<cytnx::cytnx_uint64>(lhs), self); })
    .def("__mod__", [](cytnx::UniTensor &self, const py::numpy_scalar<int32_t> &lhs) { return self.Mod(static_cast<cytnx::cytnx_int32>(lhs), self); })
    .def("__mod__", [](cytnx::UniTensor &self, const py::numpy_scalar<uint32_t> &lhs) { return self.Mod(static_cast<cytnx::cytnx_uint32>(lhs), self); })
    .def("__mod__", [](cytnx::UniTensor &self, const py::numpy_scalar<int16_t> &lhs) { return self.Mod(static_cast<cytnx::cytnx_int16>(lhs), self); })
    .def("__mod__", [](cytnx::UniTensor &self, const py::numpy_scalar<uint16_t> &lhs) { return self.Mod(static_cast<cytnx::cytnx_uint16>(lhs), self); })
    .def("__mod__", [](cytnx::UniTensor &self, const py::numpy_scalar<bool> &lhs) { return self.Mod(static_cast<cytnx::cytnx_bool>(lhs), self); })

    .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_complex128&lhs){return
    linalg::Mod(lhs,self);}) .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_complex64
    &lhs){return linalg::Mod(lhs,self);}) .def("__rmod__",[](UniTensor &self, const
    cytnx::cytnx_double    &lhs){return linalg::Mod(lhs,self);}) .def("__rmod__",[](UniTensor &self,
    const cytnx::cytnx_float     &lhs){return linalg::Mod(lhs,self);}) .def("__rmod__",[](UniTensor
    &self, const cytnx::cytnx_int64     &lhs){return linalg::Mod(lhs,self);})
    .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_uint64    &lhs){return
    linalg::Mod(lhs,self);}) .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_int32
    &lhs){return linalg::Mod(lhs,self);}) .def("__rmod__",[](UniTensor &self, const
    cytnx::cytnx_uint32    &lhs){return linalg::Mod(lhs,self);}) .def("__rmod__",[](UniTensor &self,
    const cytnx::cytnx_int16     &lhs){return linalg::Mod(lhs,self);}) .def("__rmod__",[](UniTensor
    &self, const cytnx::cytnx_uint16    &lhs){return linalg::Mod(lhs,self);})
    .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_bool      &lhs){return
    linalg::Mod(lhs,self);})
    .def("__rmod__", [](cytnx::UniTensor &self, const cytnx::Scalar &lhs) { return cytnx::linalg::Mod(lhs, self); })
    .def("__rmod__", [](cytnx::UniTensor &self, const py::numpy_scalar<std::complex<double>> &lhs) { return cytnx::linalg::Mod(static_cast<cytnx::cytnx_complex128>(lhs), self); })
    .def("__rmod__", [](cytnx::UniTensor &self, const py::numpy_scalar<std::complex<float>> &lhs) { return cytnx::linalg::Mod(static_cast<cytnx::cytnx_complex64>(lhs), self); })
    .def("__rmod__", [](cytnx::UniTensor &self, const py::numpy_scalar<double> &lhs) { return cytnx::linalg::Mod(static_cast<cytnx::cytnx_double>(lhs), self); })
    .def("__rmod__", [](cytnx::UniTensor &self, const py::numpy_scalar<float> &lhs) { return cytnx::linalg::Mod(static_cast<cytnx::cytnx_float>(lhs), self); })
    .def("__rmod__", [](cytnx::UniTensor &self, const py::numpy_scalar<int64_t> &lhs) { return cytnx::linalg::Mod(static_cast<cytnx::cytnx_int64>(lhs), self); })
    .def("__rmod__", [](cytnx::UniTensor &self, const py::numpy_scalar<uint64_t> &lhs) { return cytnx::linalg::Mod(static_cast<cytnx::cytnx_uint64>(lhs), self); })
    .def("__rmod__", [](cytnx::UniTensor &self, const py::numpy_scalar<int32_t> &lhs) { return cytnx::linalg::Mod(static_cast<cytnx::cytnx_int32>(lhs), self); })
    .def("__rmod__", [](cytnx::UniTensor &self, const py::numpy_scalar<uint32_t> &lhs) { return cytnx::linalg::Mod(static_cast<cytnx::cytnx_uint32>(lhs), self); })
    .def("__rmod__", [](cytnx::UniTensor &self, const py::numpy_scalar<int16_t> &lhs) { return cytnx::linalg::Mod(static_cast<cytnx::cytnx_int16>(lhs), self); })
    .def("__rmod__", [](cytnx::UniTensor &self, const py::numpy_scalar<uint16_t> &lhs) { return cytnx::linalg::Mod(static_cast<cytnx::cytnx_uint16>(lhs), self); })
    .def("__rmod__", [](cytnx::UniTensor &self, const py::numpy_scalar<bool> &lhs) { return cytnx::linalg::Mod(static_cast<cytnx::cytnx_bool>(lhs), self); })
    */

    // __pow__/__ipow__: UniTensor::Pow/Pow_ only take a plain double
    // exponent (same shape as Tensor's, see tensor_py.cpp), so bind
    // py::int_ and numpy_scalar<float> explicitly rather than relying on
    // pybind11's implicit double conversion, for the same #916 stub-quality
    // reason.
    .def("__pow__",
         [](UniTensor &self, const py::numpy_scalar<float> &p) {
           return self.Pow(static_cast<cytnx::cytnx_double>(static_cast<cytnx::cytnx_float>(p)));
         })
    .def("__pow__",
         [](UniTensor &self, const py::int_ &p) {
           return dispatch_pyint(
             p, [&](auto v) { return self.Pow(static_cast<cytnx::cytnx_double>(v)); });
         })
    .def("__pow__", [](UniTensor &self, const cytnx::cytnx_double &p) { return self.Pow(p); })
    .def("__ipow__",
         [](py::object self, const py::numpy_scalar<float> &p) {
           self.cast<UniTensor &>().Pow_(
             static_cast<cytnx::cytnx_double>(static_cast<cytnx::cytnx_float>(p)));
           return self;
         })
    .def("__ipow__",
         [](py::object self, const py::int_ &p) {
           dispatch_pyint(p, [&](auto v) {
             self.cast<UniTensor &>().Pow_(static_cast<cytnx::cytnx_double>(v));
           });
           return self;
         })
    .def("__ipow__",
         [](py::object self, const cytnx::cytnx_double &p) {
           self.cast<UniTensor &>().Pow_(p);
           return self;
         })
    .def("Pow", &UniTensor::Pow)
    .def("Pow_",
         [](py::object self, const double &p) {
           self.cast<UniTensor &>().Pow_(p);
           return self;
         })

    .def("Inv_",
         [](py::object self, double clip){
           self.cast<UniTensor &>().Inv_(clip);
           return self;
         },
         py::arg("clip")=-1)
    .def("Inv", [](UniTensor &self, double clip){
                            return self.Inv(clip);
                    },
                    py::arg("clip")=-1)
    .def("Conj_",
         [](py::object self) {
           self.cast<UniTensor &>().Conj_();
           return self;
         })
    .def("Conj", &UniTensor::Conj)

    .def("Trace_",
         [](py::object self, const cytnx_int64 &a, const cytnx_int64 &b) {
           self.cast<UniTensor &>().Trace_(a, b);
           return self;
         },
         py::arg("a")=0, py::arg("b")=1)

    .def("Trace_",
         [](py::object self, const std::string &a, const std::string &b) {
           self.cast<UniTensor &>().Trace_(a, b);
           return self;
         },
         py::arg("a"), py::arg("b"))


    .def("Trace", [](UniTensor &self, const cytnx_int64 &a, const cytnx_int64 &b){

                        return self.Trace(a,b);

                 },
                 py::arg("a")=0, py::arg("b")=1)

    .def("Trace", [](UniTensor &self, const std::string &a, const std::string &b){
                    return self.Trace(a,b);
                 },
                 py::arg("a"), py::arg("b"))

    // Norm() is deprecated (returns a rank-0 Tensor); use norm() (returns a python float).
    // Deprecation-warning suppression lives in unitensor_Norm_deprecated() at file scope (a
    // #pragma GCC diagnostic cannot legally sit inside this .def() chain expression under GCC).
    .def("Norm", &unitensor_Norm_deprecated)
    // norm() returns a Scalar in C++ (dtype-preserving); hand Python a native float so
    // cytnx.Scalar is not exposed on the Python surface.
    .def("norm", [](UniTensor &self) { return double(self.norm()); })
    .def("Transpose_",
         [](py::object self) {
           self.cast<UniTensor &>().Transpose_();
           return self;
         })
    .def("Transpose", &UniTensor::Transpose)
    .def("normalize_",
         [](py::object self) {
           self.cast<UniTensor &>().normalize_();
           return self;
         })
    .def("normalize", &UniTensor::normalize)

    .def("Dagger_",
         [](py::object self) {
           self.cast<UniTensor &>().Dagger_();
           return self;
         })
    .def("Dagger", &UniTensor::Dagger)
    .def("tag_",
         [](py::object self) {
           self.cast<UniTensor &>().tag_();
           return self;
         })
    .def("tag",
         [](py::object self) {
           if (PyErr_WarnEx(PyExc_DeprecationWarning, "tag() is deprecated, use tag_() instead.", 1) < 0) throw py::error_already_set();
           self.cast<UniTensor &>().tag_();
           return self;
         })
    .def("truncate",[](UniTensor &self, const cytnx_int64 &bond_idx, const cytnx_uint64 &dim){

                         return self.truncate(bond_idx, dim);

                    },
                    py::arg("bond_idx"), py::arg("dim"))
    .def("truncate",[](UniTensor &self, const std::string &label, const cytnx_uint64 &dim){
                        return self.truncate(label, dim);
                    },
                    py::arg("label"), py::arg("dim"))

    .def("truncate_",
         [](py::object self, const cytnx_int64 &bond_idx, const cytnx_uint64 &dim) {
           self.cast<UniTensor &>().truncate_(bond_idx, dim);
           return self;
         },
         py::arg("bond_idx"), py::arg("dim"))

    .def("truncate_",
         [](py::object self, const std::string &label, const cytnx_uint64 &dim) {
           self.cast<UniTensor &>().truncate_(label, dim);
           return self;
         },
         py::arg("label"), py::arg("dim"))

    //[Generator]
    .def_static("identity", [](cytnx_uint64 dim, const std::vector<std::string> &in_labels,
                  cytnx_bool is_diag,
                  unsigned int dtype,
                  int device,
                  const std::string &name)
                {
                  return UniTensor::identity(dim, in_labels, is_diag, dtype, device, name);
                }, py::arg("dim"), py::arg("labels") = std::vector<std::string>(),
                   py::arg("is_diag") = false,
                   py::arg("dtype") = (unsigned int)Type.Double,
                   py::arg("device") = int(Device.cpu),
                   py::arg("name") = std::string(""))
     .def_static("eye", [](cytnx_uint64 dim, const std::vector<std::string> &in_labels,
                  cytnx_bool is_diag,
                  unsigned int dtype,
                  int device,
                  const std::string &name)
                {
                  return UniTensor::eye(dim, in_labels, is_diag, dtype, device, name);
                }, py::arg("dim"), py::arg("labels") = std::vector<std::string>(),
                   py::arg("is_diag") = false,
                   py::arg("dtype") = (unsigned int)Type.Double,
                   py::arg("device") = int(Device.cpu),
                   py::arg("name") = std::string(""))
    .def_static("ones", [](cytnx_uint64 Nelem, const std::vector<std::string> &in_labels,
                  unsigned int dtype,
                  int device,
                  const std::string &name)
                {
                  return UniTensor::ones({Nelem}, in_labels,dtype,device,name);
                }, py::arg("Nelem"), py::arg("labels") = std::vector<std::string>(), py::arg("dtype") = (unsigned int)Type.Double,
                   py::arg("device") = int(Device.cpu),
                   py::arg("name") = std::string(""))
    .def_static("ones", [](const std::vector<cytnx_uint64> &shape, const std::vector<std::string> &in_labels,
                  unsigned int dtype,
                  int device,
                  const std::string &name)
                {
                  return UniTensor::ones(shape, in_labels,dtype,device,name);
                }, py::arg("shape"), py::arg("labels") = std::vector<std::string>(), py::arg("dtype") = (unsigned int)Type.Double,
                   py::arg("device") = int(Device.cpu),
                   py::arg("name") = std::string(""))
     .def_static("zeros", [](cytnx_uint64 Nelem, const std::vector<std::string> &in_labels,
                  unsigned int dtype,
                  int device,
                  const std::string &name)
                {
                  return UniTensor::zeros({Nelem}, in_labels,dtype,device,name);
                }, py::arg("Nelem"), py::arg("labels") = std::vector<std::string>(), py::arg("dtype") = (unsigned int)Type.Double,
                   py::arg("device") = int(Device.cpu),
                   py::arg("name") = std::string(""))
     .def_static("zeros", [](const std::vector<cytnx_uint64> &shape, const std::vector<std::string> &in_labels,
                  unsigned int dtype,
                  int device,
                  const std::string &name)
                {
                  return UniTensor::zeros(shape, in_labels,dtype,device,name);
                }, py::arg("shape"), py::arg("labels") = std::vector<std::string>(), py::arg("dtype") = (unsigned int)Type.Double,
                   py::arg("device") = int(Device.cpu),
                   py::arg("name") = std::string(""))
     .def_static("arange", [](cytnx_uint64 Nelem, const std::vector<std::string> &in_labels,
                  const std::string &name)
                {
                  return UniTensor::arange(Nelem, in_labels,name);
                }, py::arg("Nelem"), py::arg("labels") = std::vector<std::string>(),
                   py::arg("name") = std::string(""))
     .def_static("arange", [](cytnx_double start, cytnx_double end
     ,cytnx_double step, const std::vector<std::string> &in_labels,unsigned int dtype, int device,
                  const std::string &name)
                {
                  return UniTensor::arange(start,end,step, in_labels,dtype,device,name);
                }, py::arg("start"),py::arg("end"),py::arg("step")=cytnx_double(1), py::arg("labels") = std::vector<std::string>(), py::arg("dtype") = (unsigned int)Type.Double,
                   py::arg("device") = int(Device.cpu),
                   py::arg("name") = std::string(""))
     .def_static("linspace", [](cytnx_double start, cytnx_double end
     ,cytnx_uint64 Nelem,bool endpoint,const std::vector<std::string> &in_labels,unsigned int dtype, int device,
                  const std::string &name)
                {
                  return UniTensor::linspace(start,end,Nelem, endpoint, in_labels,dtype,device,name);
                }, py::arg("start"),py::arg("end"),py::arg("Nelem"),py::arg("endpoint")=true,  py::arg("labels") = std::vector<std::string>(), py::arg("dtype") = (unsigned int)Type.Double,
                   py::arg("device") = int(Device.cpu),
                   py::arg("name") = std::string(""))
     .def_static("normal", [](const cytnx_uint64 &Nelem, const double &mean, const double &std,
                              const std::vector<std::string> &in_labels,
                              int64_t &seed, const unsigned int &dtype,
							  const int &device, const std::string &name)
                {
                    if(seed==-1){
                         // If no seed argument was specified by the user
                         seed = cytnx::random::__static_random_device();
                    }
                  return UniTensor::normal(Nelem, mean, std, in_labels, seed, dtype, device, name);
                },
				py::arg("Nelem"), py::arg("mean"), py::arg("std"),
				py::arg("in_labels")=std::vector<std::string>(), py::arg("seed")= -1,
				py::arg("dtype") = (unsigned int)Type.Double, py::arg("device") = int(Device.cpu),
                py::arg("name") = std::string(""))
     .def_static("normal", [](const std::vector<cytnx_uint64> &shape, const double &mean, const double &std,
                              const std::vector<std::string> &in_labels,
                              int64_t &seed, const unsigned int &dtype,
							  const int &device, const std::string &name)
                {
                    if(seed==-1){
                         // If no seed argument was specified by the user
                         seed = cytnx::random::__static_random_device();
                    }
                  return UniTensor::normal(shape, mean, std, in_labels, seed, dtype, device, name);
                },
				py::arg("shape"), py::arg("mean"), py::arg("std"),
				py::arg("in_labels")=std::vector<std::string>(), py::arg("seed")= -1,
				py::arg("dtype") = (unsigned int)Type.Double, py::arg("device") = int(Device.cpu),
                py::arg("name") = std::string(""))
     .def_static("uniform", [](const cytnx_uint64 &Nelem, const double &low, const double &high,
                              const std::vector<std::string> &in_labels,
                              int64_t &seed, const unsigned int &dtype,
							  const int &device, const std::string &name)
                {
                    if(seed==-1){
                         // If no seed argument was specified by the user
                         seed = cytnx::random::__static_random_device();
                    }
                  return UniTensor::uniform(Nelem, low, high, in_labels, seed, dtype, device, name);
                },
				py::arg("Nelem"), py::arg("low"), py::arg("high"),
				py::arg("in_labels")=std::vector<std::string>(), py::arg("seed")= -1,
				py::arg("dtype") = (unsigned int)Type.Double, py::arg("device") = int(Device.cpu),
                py::arg("name") = std::string(""))
     .def_static("uniform", [](const std::vector<cytnx_uint64> &shape, const double &low, const double &high,
                              const std::vector<std::string> &in_labels,
                              int64_t &seed, const unsigned int &dtype,
							  const int &device, const std::string &name)
                {
                    if(seed==-1){
                         // If no seed argument was specified by the user
                         seed = cytnx::random::__static_random_device();
                    }
                  return UniTensor::uniform(shape, low, high, in_labels, seed, dtype, device, name);
                },
				py::arg("shape"), py::arg("low"), py::arg("high"),
				py::arg("in_labels")=std::vector<std::string>(), py::arg("seed")= -1,
				py::arg("dtype") = (unsigned int)Type.Double, py::arg("device") = int(Device.cpu),
                py::arg("name") = std::string(""))
     .def("normal_", [](UniTensor &self, const double &mean, const double &std,
					    int64_t &seed)
                {
                    if(seed==-1){
                         // If no seed argument was specified by the user
                         seed = cytnx::random::__static_random_device();
                    }
                  self.normal_(mean, std, seed);
                },
				py::arg("mean"), py::arg("std"), py::arg("seed")= -1)
     .def("uniform_", [](UniTensor &self, const double &low, const double &high,
					     int64_t &seed)
                {
                    if(seed==-1){
                         // If no seed argument was specified by the user
                         seed = cytnx::random::__static_random_device();
                    }
                  self.uniform_(low, high, seed);
                },
				py::arg("low"), py::arg("high"), py::arg("seed")= -1)

	.def("convert_from_",
                    [](py::object self, const UniTensor &in, bool force, cytnx_double tol) {
                      self.cast<UniTensor &>().convert_from_(in, force, tol);
                      return self;
                    },
                    py::arg("Tin"), py::arg("force") = false, py::arg("tol") = 0.)

	.def("convert_from",
                    [](py::object self, const UniTensor &in, bool force, cytnx_double tol) {
                      if (PyErr_WarnEx(PyExc_DeprecationWarning,
                        "convert_from() is deprecated, use convert_from_() instead.", 1) < 0) throw py::error_already_set();
                      self.cast<UniTensor &>().convert_from_(in, force, tol);
                      return self;
                    },
                    py::arg("Tin"), py::arg("force") = false, py::arg("tol") = 0.)

     .def("get_qindices",  [](UniTensor &self, const cytnx_uint64 &bidx){return self.get_qindices(bidx);});
  ;  // end of object line

  //   m.def("Contract", Contract, py::arg("Tl"), py::arg("Tr"), py::arg("cacheL") = false,
  //         py::arg("cacheR") = false);
  // GIL: see the guard discipline note in linalg_py.cpp
  m.def(
    "Contract",
    [](const UniTensor &inL, const UniTensor &inR, const bool &cacheL,
       const bool &cacheR) -> UniTensor { return Contract(inL, inR, cacheL, cacheR); },
    py::arg("Tl"), py::arg("Tr"), py::arg("cacheL") = false, py::arg("cacheR") = false,
    py::call_guard<py::gil_scoped_release>());
  m.def(
    "Contract",
    [](const std::vector<UniTensor> &TNs, const std::string &order,
       const bool &optimal) -> UniTensor { return Contract(TNs, order, optimal); },
    py::arg("TNs"), py::arg("order") = "", py::arg("optimal") = true,
    py::call_guard<py::gil_scoped_release>());
  m.def(
    "Contracts",
    [](const std::vector<UniTensor> &TNs, const std::string &order,
       const bool &optimal) -> UniTensor { return Contracts(TNs, order, optimal); },
    py::arg("TNs"), py::arg("order") = "", py::arg("optimal") = true,
    py::call_guard<py::gil_scoped_release>());
}
#endif
