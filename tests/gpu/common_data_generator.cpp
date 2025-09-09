#define NEED_GEN_COMMON_DATA 0
#if NEED_GEN_COMMON_DATA

/*
 * This file is about the code to generate the test data in the directory
 *   "tests/test_data_base/common"
 *   if need to re generate the data again by google test, set the macro
 *   NEED_GEN_COMMON_DATA equal 1, and generate the data by google test again.
 */

  #include "cytnx.hpp"
  #include <gtest/gtest.h>
  #include "test_tools.h"

using namespace cytnx;
using namespace testing;
using namespace TestTools;

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

  TEST(CommonDataGen, Dense_gen_nonDiag) {
    std::vector<cytnx::Bond> bonds = {cytnx::Bond(5), cytnx::Bond(6), cytnx::Bond(3)};
    cytnx::cytnx_int64 row_rank = 1;
    std::vector<std::string> labels = {};
    bool is_diag;
    for (auto dtype : dtype_list1) {
      auto UT =
        cytnx::UniTensor(bonds, labels, row_rank, dtype, cytnx::Device.cpu, is_diag = false);
      InitUniTensorUniform(UT);
      std::string file_name = dataRoot + "dense_nondiag_" + GetDTypeFileName(dtype);
      UT.Save(file_name);
    }
  }

  TEST(CommonDataGen, U1_sym_gen) {
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
      std::string file_name = dataRoot + "sym_UT_U1_" + GetDTypeFileName(dtype);
      UT.Save(file_name);
    }
  }

  TEST(CommonDataGen, Z2_sym_gen) {
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
      std::string file_name = dataRoot + "sym_UT_Z2_" + GetDTypeFileName(dtype);
      UT.Save(file_name);
    }
  }

  TEST(CommonDataGen, Z3_sym_gen) {
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
      std::string file_name = dataRoot + "sym_UT_Z3_" + GetDTypeFileName(dtype);
      UT.Save(file_name);
    }
  }

  TEST(CommonDataGen, U1xZ2_sym_gen) {
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
      std::string file_name = dataRoot + "sym_UT_U1xZ2_" + GetDTypeFileName(dtype);
      UT.Save(file_name);
    }
  }

  TEST(CommonDataGen, U1_sym_zeros_gen) {
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
    std::string file_name = dataRoot + "sym_UT_U1_zeros_F64";
    UT.Save(file_name);
  }

  TEST(CommonDataGen, U1_sym_one_elem_gen) {
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
    std::string file_name = dataRoot + "sym_UT_U1_one_elem_F64";
    UT.Save(file_name);
  }

}  // namespace CommonDataGen

#endif
