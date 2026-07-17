#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "cytnx.hpp"

namespace cytnx {
  namespace {

    // Pin the legacy accessor signature: `int &n() const` must keep returning a
    // mutable reference even from a const Symmetry (see the note in Symmetry.hpp;
    // cleanup is tracked with the #840-era API follow-ups).
    static_assert(std::is_same_v<decltype(std::declval<const Symmetry &>().n()), int &>,
                  "Symmetry::n() must preserve the legacy `int &n() const` signature");

    const std::string kDataDir = CYTNX_TEST_DATA_DIR "/common/Symmetry/";

    std::vector<char> ReadAllBytes(const std::string &path) {
      std::ifstream f(path, std::ios::binary);
      EXPECT_TRUE(f.is_open()) << "cannot open " << path;
      return std::vector<char>((std::istreambuf_iterator<char>(f)),
                               std::istreambuf_iterator<char>());
    }

    std::string TmpPath(const std::string &name) {
      return (std::filesystem::temp_directory_path() / name).string();
    }

    // Save `sym` to a temporary file and return its bytes.
    std::vector<char> SaveToBytes(const Symmetry &sym, const std::string &tag) {
      const std::string path = TmpPath("cytnx_symmetry_test_" + tag + ".cysym");
      sym.Save(path);
      std::vector<char> bytes = ReadAllBytes(path);
      std::remove(path.c_str());
      return bytes;
    }

    // ---------------------------------------------------------------------------
    // Byte-compatibility contract.
    //
    // The fixture files under tests/test_data_base/common/Symmetry/ were written
    // by Symmetry::Save()/Bond::Save() at the last commit of the
    // boost::intrusive_ptr<Symmetry_base> implementation (d6dcd160, see
    // tests/common_data_generator.cpp, CommonDataGen.Symmetry_gen).  Loading them
    // must reproduce the original objects, and re-saving freshly constructed
    // objects must reproduce the identical bytes (magic 777 : u32, stype : i32,
    // n : i32).
    // ---------------------------------------------------------------------------

    TEST(Symmetry, LoadFixtureU1) {
      Symmetry sym = Symmetry::Load(kDataDir + "sym_U1.cysym");
      EXPECT_EQ(sym.stype(), SymmetryType::U);
      EXPECT_EQ(sym.n(), 1);
      EXPECT_EQ(sym.stype_str(), "U1");
      EXPECT_FALSE(sym.is_fermionic());
      EXPECT_TRUE(sym == Symmetry::U1());
    }

    TEST(Symmetry, LoadFixtureZ3) {
      Symmetry sym = Symmetry::Load(kDataDir + "sym_Z3.cysym");
      EXPECT_EQ(sym.stype(), SymmetryType::Z);
      EXPECT_EQ(sym.n(), 3);
      EXPECT_EQ(sym.stype_str(), "Z3");
      EXPECT_FALSE(sym.is_fermionic());
      EXPECT_TRUE(sym == Symmetry::Zn(3));
      EXPECT_TRUE(sym != Symmetry::Zn(2));
    }

    TEST(Symmetry, LoadFixtureFermionParity) {
      Symmetry sym = Symmetry::Load(kDataDir + "sym_fPar.cysym");
      EXPECT_EQ(sym.stype(), SymmetryType::fPar);
      EXPECT_EQ(sym.n(), -2);
      EXPECT_EQ(sym.stype_str(), "fP");
      EXPECT_TRUE(sym.is_fermionic());
      EXPECT_TRUE(sym == Symmetry::FermionParity());
      EXPECT_EQ(sym.get_fermion_parity(0), fermionParity::EVEN);
      EXPECT_EQ(sym.get_fermion_parity(1), fermionParity::ODD);
    }

    TEST(Symmetry, LoadFixtureFermionNumber) {
      Symmetry sym = Symmetry::Load(kDataDir + "sym_fNum.cysym");
      EXPECT_EQ(sym.stype(), SymmetryType::fNum);
      EXPECT_EQ(sym.n(), -1);
      EXPECT_EQ(sym.stype_str(), "f#");
      EXPECT_TRUE(sym.is_fermionic());
      EXPECT_TRUE(sym == Symmetry::FermionNumber());
      EXPECT_EQ(sym.get_fermion_parity(4), fermionParity::EVEN);
      EXPECT_EQ(sym.get_fermion_parity(-3), fermionParity::ODD);
    }

    TEST(Symmetry, SaveIsByteIdenticalToFixture) {
      EXPECT_EQ(SaveToBytes(Symmetry::U1(), "u1"), ReadAllBytes(kDataDir + "sym_U1.cysym"));
      EXPECT_EQ(SaveToBytes(Symmetry::Zn(3), "z3"), ReadAllBytes(kDataDir + "sym_Z3.cysym"));
      EXPECT_EQ(SaveToBytes(Symmetry::FermionParity(), "fpar"),
                ReadAllBytes(kDataDir + "sym_fPar.cysym"));
      EXPECT_EQ(SaveToBytes(Symmetry::FermionNumber(), "fnum"),
                ReadAllBytes(kDataDir + "sym_fNum.cysym"));
    }

    TEST(Symmetry, BondFixtureLoadsAndResavesByteIdentical) {
      const std::string fixture = kDataDir + "bond_mixed_syms.cybd";

      // The same Bond as written by the generator: qnum rows are (U1, Z3, fPar, fNum).
      Bond fresh(
        BD_KET, {{-1, 0, 0, 2}, {0, 1, 1, 1}, {2, 2, 0, -4}}, {2, 1, 3},
        {Symmetry::U1(), Symmetry::Zn(3), Symmetry::FermionParity(), Symmetry::FermionNumber()});

      Bond loaded = Bond::Load(fixture);
      EXPECT_TRUE(loaded == fresh);
      ASSERT_EQ(loaded.Nsym(), 4);
      EXPECT_EQ(loaded.syms()[0], Symmetry::U1());
      EXPECT_EQ(loaded.syms()[1], Symmetry::Zn(3));
      EXPECT_EQ(loaded.syms()[2], Symmetry::FermionParity());
      EXPECT_EQ(loaded.syms()[3], Symmetry::FermionNumber());

      const std::string tmp = TmpPath("cytnx_symmetry_test_bond.cybd");
      fresh.Save(tmp);
      EXPECT_EQ(ReadAllBytes(tmp), ReadAllBytes(fixture));
      std::remove(tmp.c_str());
    }

    // ---------------------------------------------------------------------------
    // API pinning: construction, accessors, rules.  These exercise the public
    // surface that the value-type refactor (#842) must preserve.
    // ---------------------------------------------------------------------------

    TEST(Symmetry, DefaultConstructionIsU1) {
      Symmetry sym;
      EXPECT_EQ(sym.stype(), SymmetryType::U);
      EXPECT_EQ(sym.n(), 1);
      EXPECT_TRUE(sym == Symmetry::U1());
    }

    TEST(Symmetry, StypeNCtorMatchesFactories) {
      EXPECT_TRUE(Symmetry(SymmetryType::U, 1) == Symmetry::U1());
      EXPECT_TRUE(Symmetry(SymmetryType::Z, 4) == Symmetry::Zn(4));
      EXPECT_TRUE(Symmetry(SymmetryType::fPar) == Symmetry::FermionParity());
      EXPECT_TRUE(Symmetry(SymmetryType::fNum) == Symmetry::FermionNumber());
    }

    TEST(Symmetry, InvalidConstructionThrows) {
      EXPECT_THROW(Symmetry::Zn(1), std::logic_error);
      EXPECT_THROW(Symmetry::Zn(0), std::logic_error);
      EXPECT_THROW(Symmetry{SymmetryType::Void}, std::logic_error);
      EXPECT_THROW(Symmetry{42}, std::logic_error);
    }

    TEST(Symmetry, CloneAndCopyCompareEqual) {
      Symmetry z5 = Symmetry::Zn(5);
      Symmetry cloned = z5.clone();
      Symmetry copied = z5;
      EXPECT_TRUE(cloned == z5);
      EXPECT_TRUE(copied == z5);
      EXPECT_EQ(cloned.stype_str(), "Z5");
      Symmetry reassigned;
      reassigned = z5;
      EXPECT_TRUE(reassigned == z5);
    }

    TEST(Symmetry, EqualityIsStypeAndN) {
      EXPECT_TRUE(Symmetry::Zn(2) == Symmetry::Zn(2));
      EXPECT_TRUE(Symmetry::Zn(2) != Symmetry::Zn(3));
      EXPECT_TRUE(Symmetry::U1() != Symmetry::Zn(2));
      EXPECT_TRUE(Symmetry::FermionParity() != Symmetry::FermionNumber());
      EXPECT_TRUE(Symmetry::U1() != Symmetry::FermionNumber());
    }

    TEST(Symmetry, CheckQnumRanges) {
      EXPECT_TRUE(Symmetry::U1().check_qnum(-7));
      EXPECT_TRUE(Symmetry::U1().check_qnums({-1, 0, 99}));

      Symmetry z3 = Symmetry::Zn(3);
      EXPECT_TRUE(z3.check_qnum(0));
      EXPECT_TRUE(z3.check_qnum(2));
      EXPECT_FALSE(z3.check_qnum(3));
      EXPECT_FALSE(z3.check_qnum(-1));
      EXPECT_TRUE(z3.check_qnums({0, 1, 2}));
      EXPECT_FALSE(z3.check_qnums({0, 3}));

      Symmetry fpar = Symmetry::FermionParity();
      EXPECT_TRUE(fpar.check_qnum(0));
      EXPECT_TRUE(fpar.check_qnum(1));
      EXPECT_FALSE(fpar.check_qnum(2));
      EXPECT_FALSE(fpar.check_qnum(-1));

      EXPECT_TRUE(Symmetry::FermionNumber().check_qnum(-5));
      EXPECT_TRUE(Symmetry::FermionNumber().check_qnums({-2, 0, 7}));
    }

    TEST(Symmetry, CombineAndReverseRulesAllKinds) {
      // U1: plain addition, reverse = negation.
      Symmetry u1 = Symmetry::U1();
      EXPECT_EQ(u1.combine_rule(2, 3), 5);
      EXPECT_EQ(u1.combine_rule(2, 3, /*is_reverse=*/true), -5);
      EXPECT_EQ(u1.reverse_rule(-4), 4);

      // Zn: modular addition.
      Symmetry z3 = Symmetry::Zn(3);
      EXPECT_EQ(z3.combine_rule(2, 2), 1);
      EXPECT_EQ(z3.combine_rule(1, 1, /*is_reverse=*/true), 1);
      EXPECT_EQ(z3.reverse_rule(2), 1);

      // FermionParity: mod-2 addition.
      Symmetry fpar = Symmetry::FermionParity();
      EXPECT_EQ(fpar.combine_rule(1, 1), 0);
      EXPECT_EQ(fpar.combine_rule(1, 0), 1);
      EXPECT_EQ(fpar.reverse_rule(1), 1);
      EXPECT_EQ(fpar.reverse_rule(0), 2);  // legacy: -0 + 2 (not canonicalized)

      // FermionNumber: plain addition, reverse = negation.
      Symmetry fnum = Symmetry::FermionNumber();
      EXPECT_EQ(fnum.combine_rule(2, -5), -3);
      EXPECT_EQ(fnum.reverse_rule(3), -3);
    }

    TEST(Symmetry, CombineRuleVectorForms) {
      Symmetry u1 = Symmetry::U1();
      std::vector<cytnx_int64> outv =
        u1.combine_rule(std::vector<cytnx_int64>{0, 1}, std::vector<cytnx_int64>{10, 20});
      EXPECT_EQ(outv, (std::vector<cytnx_int64>{10, 20, 11, 21}));

      Symmetry z3 = Symmetry::Zn(3);
      outv = z3.combine_rule(std::vector<cytnx_int64>{1, 2}, std::vector<cytnx_int64>{2});
      EXPECT_EQ(outv, (std::vector<cytnx_int64>{0, 1}));

      Symmetry fpar = Symmetry::FermionParity();
      outv = fpar.combine_rule(std::vector<cytnx_int64>{0, 1}, std::vector<cytnx_int64>{1});
      EXPECT_EQ(outv, (std::vector<cytnx_int64>{1, 0}));

      Symmetry fnum = Symmetry::FermionNumber();
      outv = fnum.combine_rule(std::vector<cytnx_int64>{-1, 2}, std::vector<cytnx_int64>{1});
      EXPECT_EQ(outv, (std::vector<cytnx_int64>{0, 3}));
    }

    TEST(Symmetry, OutParamForms) {
      Symmetry z4 = Symmetry::Zn(4);

      cytnx_int64 out_scalar = -777;
      z4.combine_rule_(out_scalar, 3, 2);
      EXPECT_EQ(out_scalar, 1);
      z4.combine_rule_(out_scalar, 3, 2, /*is_reverse=*/true);
      EXPECT_EQ(out_scalar, 3);

      std::vector<cytnx_int64> out_vec;
      z4.combine_rule_(out_vec, {0, 3}, {1, 2});
      EXPECT_EQ(out_vec, (std::vector<cytnx_int64>{1, 2, 0, 1}));

      z4.reverse_rule_(out_scalar, 1);
      EXPECT_EQ(out_scalar, 3);
    }

    TEST(Symmetry, FermionParityInvalidQnumThrows) {
      EXPECT_THROW(Symmetry::FermionParity().get_fermion_parity(5), std::logic_error);
    }

    // The legacy check_qnums quirk (always false for non-empty input; compared
    // against the n = -2 stype sentinel) was fixed on master by #1012; the two
    // tests below carry over master's coverage of the fixed behavior, and the
    // variant implementation delegates per-element to check_qnum accordingly.
    TEST(Symmetry, FermionParityCheckQnum) {
      Symmetry sym = Symmetry::FermionParity();
      EXPECT_TRUE(sym.check_qnum(0));
      EXPECT_TRUE(sym.check_qnum(1));
      EXPECT_FALSE(sym.check_qnum(-1));
      EXPECT_FALSE(sym.check_qnum(2));
    }

    TEST(Symmetry, FermionParityCheckQnums) {
      Symmetry sym = Symmetry::FermionParity();
      EXPECT_TRUE(sym.check_qnums({0, 1}));
      EXPECT_FALSE(sym.check_qnums({2}));
      EXPECT_FALSE(sym.check_qnums({-1}));
      EXPECT_FALSE(sym.check_qnums({0, 1, 2}));
    }

    TEST(Symmetry, FermionParityCheckQnumsEmptyIsVacuouslyTrue) {
      Symmetry fpar = Symmetry::FermionParity();
      EXPECT_TRUE(fpar.check_qnums({}));
    }

    TEST(Symmetry, LoadRejectsBadMagic) {
      const std::string path = TmpPath("cytnx_symmetry_test_bad_magic.cysym");
      {
        std::ofstream f(path, std::ios::binary);
        const unsigned int bad_magic = 778;  // anything but 777
        const int stype = SymmetryType::U;
        const int n = 1;
        f.write(reinterpret_cast<const char *>(&bad_magic), sizeof(bad_magic));
        f.write(reinterpret_cast<const char *>(&stype), sizeof(stype));
        f.write(reinterpret_cast<const char *>(&n), sizeof(n));
      }
      EXPECT_THROW(Symmetry::Load(path), std::logic_error);
      std::remove(path.c_str());
    }

    TEST(Symmetry, BosonicKindsReportEvenParity) {
      EXPECT_EQ(Symmetry::U1().get_fermion_parity(3), fermionParity::EVEN);
      EXPECT_EQ(Symmetry::Zn(2).get_fermion_parity(1), fermionParity::EVEN);
    }

    TEST(Symmetry, SaveLoadRoundTripThroughFreshFile) {
      const Symmetry originals[] = {Symmetry::U1(), Symmetry::Zn(7), Symmetry::FermionParity(),
                                    Symmetry::FermionNumber()};
      int idx = 0;
      for (const Symmetry &sym : originals) {
        const std::string path =
          TmpPath("cytnx_symmetry_roundtrip_" + std::to_string(idx++) + ".cysym");
        sym.Save(path);
        Symmetry back = Symmetry::Load(path);
        EXPECT_TRUE(back == sym);
        EXPECT_EQ(back.stype_str(), sym.stype_str());
        EXPECT_EQ(back.is_fermionic(), sym.is_fermionic());
        std::remove(path.c_str());
      }
    }
  }  // namespace
}  // namespace cytnx
