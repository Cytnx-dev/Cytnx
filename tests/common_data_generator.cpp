
namespace cytnx {
  namespace {

#define NEED_GEN_COMMON_DATA 0
#if NEED_GEN_COMMON_DATA

    /*
     * This file contains code to generate the test data in the directory
     *   "tests/test_data_base/common".
     *   To re-generate the data again by google test, set the macro
     *   NEED_GEN_COMMON_DATA equal 1.
     */

  #include "cytnx.hpp"
  #include <gtest/gtest.h>
  #include "test_tools.h"
    namespace CommonDataGen {

      std::string dataRoot = CYTNX_TEST_DATA_DIR "/common/";
      static std::vector<unsigned int> dtype_list1 = {
        Type.ComplexDouble,
        Type.Double,
      };

      std::string GetDTypeFileName(unsigned int dtype) {
        switch (dtype) {
          case Type.ComplexDouble:
            return "C128";
          case Type.ComplexFloat:
            return "C64";
          case Type.Double:
            return "F64";
          case Type.Float:
            return "F32";
          case Type.Int64:
            return "I64";
          case Type.Uint64:
            return "U64";
          case Type.Int32:
            return "I32";
          case Type.Uint32:
            return "U32";
          case Type.Int16:
            return "I16";
          case Type.Uint16:
            return "U16";
          case Type.Bool:
            return "Bool";
          default:
            assert(false);
        }
      }

      TEST(CommonDataGen, DenseGenNonDiag) {
        std::vector<Bond> bonds = {Bond(5), Bond(6), Bond(3)};
        cytnx_int64 row_rank = 1;
        std::vector<std::string> labels = {};
        bool is_diag;
        for (auto dtype : dtype_list1) {
          auto UT = UniTensor(bonds, labels, row_rank, dtype, Device.cpu, is_diag = false);
          InitUniTensorUniform(UT);
          std::string file_name = dataRoot + "dense_nondiag_" + GetDTypeFileName(dtype) + ".cytnx";
          UT.Save(file_name);
        }
      }

      TEST(CommonDataGen, U1SymGen) {
        // construct bonds
        std::vector<std::vector<cytnx_int64>> qnums1 = {{0}, {1}, {0}, {1}, {2}};
        std::vector<cytnx_uint64> degs = {1, 2, 3, 4, 5};
        auto syms = std::vector<Symmetry>(qnums1[0].size(), Symmetry(SymmetryType::U));
        auto bond_ket = Bond(BD_KET, qnums1, degs, syms);
        std::vector<std::vector<cytnx_int64>> qnums2 = {{-1}, {-1}, {0}, {2}, {1}};
        syms = std::vector<Symmetry>(qnums2[0].size(), Symmetry(SymmetryType::U));
        auto bond_bra = Bond(BD_BRA, qnums2, degs, syms);
        std::vector<Bond> bonds = {bond_ket, bond_ket, bond_bra, bond_bra};
        for (auto dtype : dtype_list1) {
          if (dtype == Type.Bool)  // It will throw error from Hstack.
            continue;
          cytnx_int64 row_rank = -1;
          std::vector<std::string> labels = {};
          // currently only for cpu device
          auto UT = UniTensor(bonds, labels, row_rank, dtype, Device.cpu, false);
          InitUniTensorUniform(UT);
          std::string file_name = dataRoot + "sym_UT_U1_" + GetDTypeFileName(dtype) + ".cytnx";
          UT.Save(file_name);
        }
      }

      TEST(CommonDataGen, Z2SymGen) {
        // construct bonds
        std::vector<std::vector<cytnx_int64>> qnums1 = {{0}, {1}, {0}, {1}};
        std::vector<cytnx_uint64> degs = {1, 2, 3, 4};
        int n;  // Zn
        auto syms = std::vector<Symmetry>({Symmetry(SymmetryType::Z, n = 2)});  // Z2
        auto bond_ket = Bond(BD_KET, qnums1, degs, syms);
        std::vector<std::vector<cytnx_int64>> qnums2 = {{0}, {1}, {1}};
        degs = {2, 1, 3};
        auto bond_bra = Bond(BD_BRA, qnums2, degs, syms);
        std::vector<Bond> bonds = {bond_ket, bond_ket, bond_bra, bond_bra};
        // test
        for (auto dtype : dtype_list1) {
          if (dtype == Type.Bool)  // It will throw error from Hstack.
            continue;
          cytnx_int64 row_rank = -1;
          std::vector<std::string> labels = {};
          // currently only for cpu device
          auto UT = UniTensor(bonds, labels, row_rank, dtype, Device.cpu, false);
          InitUniTensorUniform(UT);
          std::string file_name = dataRoot + "sym_UT_Z2_" + GetDTypeFileName(dtype) + ".cytnx";
          UT.Save(file_name);
        }
      }

      TEST(CommonDataGen, Z3SymGen) {
        // construct bonds
        std::vector<std::vector<cytnx_int64>> qnums1 = {{0}, {1}, {0}, {2}};
        std::vector<cytnx_uint64> degs = {1, 2, 3, 4};
        int n;  // Zn
        auto syms = std::vector<Symmetry>({Symmetry(SymmetryType::Z, n = 3)});  // Z3
        auto bond_ket = Bond(BD_KET, qnums1, degs, syms);
        std::vector<std::vector<cytnx_int64>> qnums2 = {{2}, {1}, {1}};
        degs = {2, 1, 3};
        auto bond_bra = Bond(BD_BRA, qnums2, degs, syms);
        std::vector<Bond> bonds = {bond_ket, bond_ket, bond_bra, bond_bra};
        // test
        for (auto dtype : dtype_list1) {
          if (dtype == Type.Bool)  // It will throw error from Hstack.
            continue;
          cytnx_int64 row_rank = -1;
          std::vector<std::string> labels = {};
          // currently only for cpu device
          auto UT = UniTensor(bonds, labels, row_rank, dtype, Device.cpu, false);
          InitUniTensorUniform(UT);
          std::string file_name = dataRoot + "sym_UT_Z3_" + GetDTypeFileName(dtype) + ".cytnx";
          UT.Save(file_name);
        }
      }

      TEST(CommonDataGen, U1xZ2SymGen) {
        // construct bonds
        std::vector<std::vector<cytnx_int64>> qnums1 = {{0, 1}, {1, 0}, {0, 0}, {2, 1}};
        std::vector<cytnx_uint64> degs = {1, 2, 3, 4};
        int n;  // Zn
        auto syms = std::vector<Symmetry>(  // U1xZ2
          {Symmetry(SymmetryType::U), Symmetry(SymmetryType::Z, n = 2)});
        auto bond_ket = Bond(BD_KET, qnums1, degs, syms);
        std::vector<std::vector<cytnx_int64>> qnums2 = {{2, 0}, {1, 1}, {1, 1}};
        degs = {2, 1, 3};
        auto bond_bra = Bond(BD_BRA, qnums2, degs, syms);
        std::vector<Bond> bonds = {bond_ket, bond_ket, bond_bra, bond_bra};
        // test
        for (auto dtype : dtype_list) {
          if (dtype == Type.Bool)  // It will throw error from Hstack.
            continue;
          cytnx_int64 row_rank = -1;
          std::vector<std::string> labels = {};
          // currently only for cpu device
          auto UT = UniTensor(bonds, labels, row_rank, dtype, Device.cpu, false);
          InitUniTensorUniform(UT);
          std::string file_name = dataRoot + "sym_UT_U1xZ2_" + GetDTypeFileName(dtype) + ".cytnx";
          UT.Save(file_name);
        }
      }

      TEST(CommonDataGen, U1SymZerosGen) {
        std::vector<std::vector<cytnx_int64>> qnums1 = {{0}, {1}, {0}, {2}};
        std::vector<cytnx_uint64> degs = {1, 2, 3, 4};
        auto syms = std::vector<Symmetry>({Symmetry(SymmetryType::U)});  // U1
        auto bond_ket = Bond(BD_KET, qnums1, degs, syms);
        std::vector<std::vector<cytnx_int64>> qnums2 = {{2}, {1}, {1}};
        degs = {2, 1, 3};
        auto bond_bra = Bond(BD_BRA, qnums2, degs, syms);
        std::vector<Bond> bonds = {bond_ket, bond_ket, bond_bra};
        cytnx_int64 row_rank = -1;
        std::vector<std::string> labels = {};
        auto UT = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, false);
        // here since we don't init elements. All block will be zeros.
        std::string file_name = dataRoot + "sym_UT_U1_zeros_F64.cytnx";
        ;
        UT.Save(file_name);
      }

      TEST(CommonDataGen, U1SymOneElemGen) {
        std::vector<std::vector<cytnx_int64>> qnums1 = {{0}};
        std::vector<cytnx_uint64> degs = {1};
        auto syms = std::vector<Symmetry>({Symmetry(SymmetryType::U)});  // U1
        auto bond_ket = Bond(BD_KET, qnums1, degs, syms);
        std::vector<std::vector<cytnx_int64>> qnums2 = {{0}};
        degs = {1};
        auto bond_bra = Bond(BD_BRA, qnums2, degs, syms);
        std::vector<Bond> bonds = {bond_ket, bond_ket, bond_bra};
        cytnx_int64 row_rank = -1;
        std::vector<std::string> labels = {};
        auto UT = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, false);
        UT.at({0, 0, 0}) = 10.5;
        std::string file_name = dataRoot + "sym_UT_U1_one_elem_F64.cytnx";
        UT.Save(file_name);
      }

      // Serialization fixtures for tests/Symmetry_test.cpp (byte-compatibility contract).
      // The committed files under test_data_base/common/Symmetry/ were generated at the
      // last commit of the boost::intrusive_ptr<Symmetry_base> implementation; the
      // value-type (std::variant) Symmetry must keep reading AND writing these exact
      // bytes (magic 777 : u32, stype_id : i32, n : i32).
      TEST(CommonDataGen, SymmetryGen) {
        const std::string symDir = dataRoot + "Symmetry/";
        Symmetry::U1().Save(symDir + "sym_U1.cysym");
        Symmetry::Zn(3).Save(symDir + "sym_Z3.cysym");
        Symmetry::FermionParity().Save(symDir + "sym_fPar.cysym");
        Symmetry::FermionNumber().Save(symDir + "sym_fNum.cysym");

        // A Bond carrying several symmetries: qnum rows are (U1, Z3, fPar, fNum).
        Bond bd(
          BD_KET, {{-1, 0, 0, 2}, {0, 1, 1, 1}, {2, 2, 0, -4}}, {2, 1, 3},
          {Symmetry::U1(), Symmetry::Zn(3), Symmetry::FermionParity(), Symmetry::FermionNumber()});
        bd.Save(symDir + "bond_mixed_syms.cybd");
      }

    }  // namespace CommonDataGen

#endif

  }  // namespace
}  // namespace cytnx
