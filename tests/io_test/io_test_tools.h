#ifndef CYTNX_TESTS_IO_TEST_TOOLS_H_
#define CYTNX_TESTS_IO_TEST_TOOLS_H_

#include <cstdio>
#include <filesystem>
#include <string>
#include <variant>

#include <gtest/gtest.h>

#include "cytnx.hpp"
#include "io.hpp"
#include "test_tools.h"

// Shared helpers for the cytnx::io HDF5 Save/Load tests. Saving and loading go
// through io::Save / io::Load only; io::open provides the file handle, which is
// passed directly as the container (an H5File is a valid H5::Group).
namespace cytnx {
  namespace IOTest {

    // Directory holding the committed reference data produced by the generator
    // in developer_tools/io_data_generator.cpp.
    inline std::string ref_data_dir() { return std::string(CYTNX_TEST_DATA_DIR) + "/io/"; }

    // Unique temp file path ending in ".h5", removed again on destruction.
    class TempH5File {
     public:
      TempH5File() : path_(std::string(std::tmpnam(nullptr)) + ".h5") {}
      ~TempH5File() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
      }
      const std::string& str() const { return path_; }

     private:
      std::string path_;
    };

    // Save a single object into a freshly truncated file.
    template <typename T>
    void SaveToFile(const T& object, const std::string& fname, const std::string& name = "obj",
                    const std::string& path = "") {
      H5::H5File file = io::open(fname, io::ACC_TRUNC);
      io::Save(object, file, name, path);
      file.close();
    }

    // Load an object of type T from a file.
    template <typename T>
    T LoadFromFile(const std::string& fname, const std::string& name = "obj",
                   const std::string& path = "") {
      io::savable_class holder = T();
      H5::H5File file = io::open(fname, io::ACC_IN);
      io::Load(holder, file, name, path);
      file.close();
      return std::get<T>(holder);
    }

    // Load an object of type T, choosing whether to restore the stored device.
    // Only valid for the loadable_to_device alternatives (Storage, Tensor, UniTensor, MPS).
    template <typename T>
    T LoadFromFileDevice(const std::string& fname, bool restore_device,
                         const std::string& name = "obj", const std::string& path = "") {
      io::loadable_to_device holder = T();
      H5::H5File file = io::open(fname, io::ACC_IN);
      io::Load(holder, file, name, path, restore_device);
      file.close();
      return std::get<T>(holder);
    }

    // Save then load through a temporary file and return the loaded copy.
    template <typename T>
    T RoundTrip(const T& object) {
      TempH5File tmp;
      SaveToFile(object, tmp.str());
      return LoadFromFile<T>(tmp.str());
    }

    // Deterministic reference objects shared by the data generator
    // (developer_tools side) and the load-from-reference tests, so both build
    // exactly the same object. Each reference object is stored in its own file
    // under the name "obj" in ref_data_dir().
    namespace ref {

      inline Storage storage() {
        std::vector<double> v(12);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<double>(i % 5);
        return Storage(v).astype(Type.ComplexDouble);
      }

      inline Tensor tensor() {
        Tensor t = Tensor({3, 4, 2}, Type.Double);
        TestTools::InitTensorUniform(t, 1234);
        return t;
      }

      inline Bond bond() {
        return Bond(BD_KET, {{0, 2}, {1, 5}, {1, 6}, {0, 1}}, {4, 7, 2, 3},
                    {Symmetry::Zn(2), Symmetry::U1()});
      }

      inline Symmetry symmetry() { return Symmetry::Zn(3); }

      inline Symmetry symmetry_fpar() { return Symmetry::FermionParity(); }

      inline Bond bond_fermionic() {
        return Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 3}, {Symmetry::FermionParity()});
      }

      inline UniTensor unitensor_dense() {
        UniTensor ut =
          UniTensor({Bond(3), Bond(4), Bond(2)}, {"a", "b", "c"}, 1, Type.Double).set_name("ref_dense");
        TestTools::InitUniTensorUniform(ut, 5678);
        return ut;
      }

      inline UniTensor unitensor_diag() {
        UniTensor ut =
          UniTensor({Bond(5), Bond(5)}, {"i", "j"}, 1, Type.Double, Device.cpu, /*is_diag=*/true)
            .set_name("ref_diag");
        TestTools::InitUniTensorUniform(ut, 8765);
        return ut;
      }

      inline UniTensor unitensor_sym() {
        Bond bk = Bond(BD_KET, {Qs(0) >> 2, Qs(1) >> 3, Qs(-1) >> 1});
        Bond bb = Bond(BD_BRA, {Qs(0) >> 2, Qs(1) >> 3, Qs(-1) >> 1});
        UniTensor ut = UniTensor({bk, bb}, {"a", "b"}, -1, Type.ComplexDouble).set_name("ref_sym");
        TestTools::InitUniTensorUniform(ut, 4321);
        return ut;
      }

      inline UniTensor unitensor_fermionic() {
        Bond bk = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 3}, {Symmetry::FermionParity()});
        Bond bb = Bond(BD_OUT, {Qs(0) >> 2, Qs(1) >> 3}, {Symmetry::FermionParity()});
        UniTensor ut = UniTensor({bk, bb}, {"a", "b"}, -1, Type.ComplexDouble).set_name("ref_ferm");
        TestTools::InitUniTensorUniform(ut, 1357);
        return ut;
      }

    }  // namespace ref

  }  // namespace IOTest
}  // namespace cytnx

#endif  // CYTNX_TESTS_IO_TEST_TOOLS_H_
