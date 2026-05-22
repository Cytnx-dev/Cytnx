#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <vector>

#include "cytnx.hpp"
#include "complex.h"
#include "H5Cpp.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace cytnx;

#ifdef BACKEND_TORCH
#else

void io_binding(py::module &m) {
  // [Submodule io]
  pybind11::module m_io = m.def_submodule("io", "Input/Output related.");

  py::enum_<cytnx::io::IoMode>(m_io, "IoMode")
    .value("ACC_TRUNC", cytnx::io::IoMode::ACC_TRUNC)
    .value("ACC_NOREPLACE", cytnx::io::IoMode::ACC_NOREPLACE)
    .value("ACC_IN", cytnx::io::IoMode::ACC_IN)
    .value("ACC_INOUT", cytnx::io::IoMode::ACC_INOUT)
    .export_values();

  // H5Object is an abstract class for all H5 objects that can contain attributes
  py::class_<H5::H5Object, std::unique_ptr<H5::H5Object, py::nodelete>>(m_io, "H5Object")
    // Attribute management
    .def(
      "attrExists",
      [](const H5::H5Object &self, const std::string &name) { return self.attrExists(name); },
      py::arg("name"))
    .def(
      "removeAttr",
      [](const H5::H5Object &self, const std::string &name) { self.removeAttr(name); },
      py::arg("name"))
    .def(
      "renameAttr",
      [](const H5::H5Object &self, const std::string &old_n, const std::string &new_n) {
        self.renameAttr(old_n, new_n);
      },
      py::arg("old_name"), py::arg("new_name"));  // end of object line

  py::class_<H5::Group, H5::H5Object>(m_io, "Group")
    // construction
    .def(py::init<>())
    .def(py::init<const H5::Group &>(), py::arg("original"))
    // Group specific methods
    .def("assign", &H5::Group::operator=)
    .def("close", &H5::Group::close)
    .def("fromClass", &H5::Group::fromClass)
    .def("getId", &H5::Group::getId)

    // inherited
    // Group management
    .def(
      "createGroup",
      [](const H5::Group &self, const std::string &name, size_t size_hint) {
        return self.createGroup(name, size_hint);
      },
      py::arg("name"), py::arg("size_hint") = 0)
    .def(
      "openGroup",
      [](const H5::Group &self, const std::string &name) { return self.openGroup(name); },
      py::arg("name"))

    // link/object management
    .def(
      "moveLink",
      [](const H5::Group &self, const std::string &src, const std::string &dst) {
        self.moveLink(src, dst);
      },
      py::arg("src_name"), py::arg("dst_name"))
    .def(
      "nameExists",
      [](const H5::Group &self, const std::string &name) { return self.nameExists(name); },
      py::arg("name"))
    .def(
      "link",
      [](const H5::Group &self, const std::string &curr, const std::string &next) {
        self.link(curr, next);
      },
      py::arg("curr_name"), py::arg("new_name"))
    .def(
      "unlink", [](const H5::Group &self, const std::string &name) { self.unlink(name); },
      py::arg("name"))

    // comments
    .def(
      "setComment",
      [](const H5::Group &self, const std::string &name, const std::string &comment) {
        self.setComment(name, comment);
      },
      py::arg("name"), py::arg("comment"))
    .def(
      "getComment",
      [](const H5::Group &self, const std::string &name) { return self.getComment(name); },
      py::arg("name"))
    .def(
      "removeComment",
      [](const H5::Group &self, const std::string &name) { self.removeComment(name); },
      py::arg("name"))

    // helper
    .def("getFileName",
         [](const H5::Group &self) { return self.getFileName(); });  // end of object line

  py::class_<H5::H5File, H5::Group>(m_io, "H5File")
    // construction
    .def(py::init<>())
    .def(py::init<const H5::H5File &>(), py::arg("original"))
    .def(py::init<const std::string &, unsigned int>(), py::arg("name"), py::arg("flags"))
    // methods
    .def("close", &H5::H5File::close)
    .def("fromClass", &H5::H5File::fromClass)
    .def("getId", &H5::H5File::getId)
    .def("getFileSize", &H5::H5File::getFileSize)
    .def("getFreeSpace", &H5::H5File::getFreeSpace)
    .def(
      "isAccessible",
      [](H5::H5File &self, const std::string &name) { return self.isAccessible(name); },
      py::arg("name"))
    .def(
      "isHdf5", [](H5::H5File &self, const std::string &name) { return self.isHdf5(name); },
      py::arg("name"))
    .def("getObjCount", [](H5::H5File &self) { return self.getObjCount(); });  // end of object line

  // implementations from cytnx::io
  m_io.def(
    "create_group",
    [](H5::Group &container, const std::string &path, bool recursive) {
      return cytnx::io::create_group(container, path, recursive);
    },
    py::arg("container"), py::arg("path"), py::arg("recursive") = true);

  m_io.def(
    "open",
    [](const std::filesystem::path &fname, cytnx::io::IoMode mode) {
      return cytnx::io::open(fname, mode);
    },
    py::arg("fname"), py::arg("mode") = cytnx::io::ACC_TRUNC);
  // m_io.def("Close", [](H5::H5File file) { cytnx::io::Close(file); }, py::arg("file"));

  m_io.def("Save", &cytnx::io::Save, py::arg("object"), py::arg("container"), py::arg("name"),
           py::arg("path") = "", py::arg("overwrite") = false);

  m_io.def(
    "c_Load",
    [](cytnx::io::savable_class &object, H5::Group &container, const std::string &name,
       const std::string &path) { cytnx::io::Load(object, container, name, path); },
    py::arg("object"), py::arg("container"), py::arg("name"), py::arg("path") = "");
  m_io.def(
    "c_Load",
    [](cytnx::io::loadable_to_device &object, H5::Group &container, const std::string &name,
       const std::string &path,
       bool restore_device) { cytnx::io::Load(object, container, name, path, restore_device); },
    py::arg("object"), py::arg("container"), py::arg("name"), py::arg("path"),
    py::arg("restore_device"));

}  // io_binding

#endif
