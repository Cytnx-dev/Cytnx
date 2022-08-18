#include "Bond_test.h"

TEST(Bond, EmptyBond) {
  Bond bd;

  EXPECT_EQ(bd.type(), BD_REG);

  EXPECT_EQ(bd.dim(), 0);
}

TEST(Bond, SimpleBondNoSymm) {
  Bond bd(5);
  EXPECT_EQ(bd.type(), BD_REG);
  EXPECT_EQ(bd.dim(), 5);
  EXPECT_EQ(bd.Nsym(), 0);
  EXPECT_THAT(bd.syms(), ::testing::ElementsAre());
  Bond bd1 = bd.redirect();
  EXPECT_EQ(bd1.type(), BD_REG);
}

TEST(Bond, SimpleBondOperation) {
  Bond bd(4, BD_KET);
  EXPECT_EQ(bd.type(), BD_KET);
  bd.set_type(BD_BRA);
  EXPECT_EQ(bd.type(), BD_BRA);

  Bond bd1 = bd.retype(BD_KET);
  EXPECT_EQ(bd1.type(), BD_KET);
  Bond bd2 = bd1.redirect();
  EXPECT_EQ(bd2.type(), BD_BRA);
}

TEST(Bond, CombineBondNoSymmReg) {
  Bond bd1(3), bd2(2);
  bd2 = bd1.combineBond(bd2);
  EXPECT_EQ(bd1.dim(), 3);
  EXPECT_EQ(bd1.type(), BD_REG);
  EXPECT_EQ(bd1.Nsym(), 0);
  EXPECT_EQ(bd2.dim(), 6);
  EXPECT_EQ(bd2.type(), BD_REG);
  EXPECT_EQ(bd2.Nsym(), 0);
}

TEST(Bond, CombineBondNoSymmBraKet) {
  Bond bd1(3, BD_BRA), bd2(2, BD_BRA);  // Bra bonds
  Bond bd3(7, BD_KET), bd4(5, BD_BRA);
  EXPECT_THROW(Bond bd5 = bd1.combineBond(bd3);, std::logic_error);
  Bond bd5 = bd1.combineBond(bd2);
  EXPECT_EQ(bd5.type(), BD_BRA);
  bd3.set_type(BD_BRA);
  bd4.set_type(BD_BRA);
  Bond bd_all = bd1.combineBonds({bd2, bd3, bd4});
  EXPECT_EQ(bd_all.dim(), 210);
  EXPECT_EQ(bd_all.type(), BD_BRA);
  EXPECT_EQ(bd_all.Nsym(), 0);
}

TEST(Bond, CombindBondSymm) {
  Bond bd_sym_a = Bond(3, BD_BRA, {{0, 1}, {2, 0}, {-4, 1}}, {Symmetry::U1(), Symmetry::Zn(2)});

  Bond bd_sym_b =
    Bond(4, BD_BRA, {{0, 0}, {2, 1}, {-1, 1}, {3, 0}}, {Symmetry::U1(), Symmetry::Zn(2)});
  Bond bd_sym_c =
    Bond(5, BD_BRA, {{1, 1}, {1, 1}, {-1, 1}, {-2, 0}, {0, 0}}, {Symmetry::U1(), Symmetry::Zn(2)});
  Bond bd_sym_d = bd_sym_a.combineBonds({bd_sym_b, bd_sym_c});
  EXPECT_EQ(bd_sym_d.type(), BD_BRA);
}
// TEST(Bond, ConstructorTypeQnums){
//   // Bond(bondType tp, const std::vector<Qnum>& qnums);
//
//   Qnum q1(1);
//   Qnum q0(0);
//   Qnum q_1(-1);
//   // Create an array of Qnums for the states of a bond.
//   std::vector<uni10::Qnum> qnums;
//   qnums.push_back(q1);
//   qnums.push_back(q1);
//   qnums.push_back(q0);
//   qnums.push_back(q0);
//   qnums.push_back(q0);
//   qnums.push_back(q_1);
//
//   // Constrcut Bond with Qnum array
//   Bond bd(uni10::BD_OUT, qnums);
//
//   // test dim
//   EXPECT_EQ(6,bd.dim());
//
//   // test Qlist
//   std::vector<uni10::Qnum> qlist = bd.Qlist();
//   EXPECT_EQ(q1,qlist[0]);
//   EXPECT_EQ(q1,qlist[1]);
//   EXPECT_EQ(q0,qlist[2]);
//   EXPECT_EQ(q0,qlist[3]);
//   EXPECT_EQ(q0,qlist[4]);
//   EXPECT_EQ(q_1,qlist[5]);
//
//   // test degeneracy
//   std::map<Qnum, int> degs = bd.degeneracy();
//
//   std::map<Qnum,int>::const_iterator it=degs.begin();
//   EXPECT_EQ(q_1, it->first);
//   EXPECT_EQ(1, it->second);
//
//   ++it;
//   EXPECT_EQ(q0,it->first);
//   EXPECT_EQ(3,it->second);
//
//   ++it;
//   EXPECT_EQ(q1,it->first);
//   EXPECT_EQ(2,it->second);
//
// }
//
// TEST(Bond, CopyConstructor){
//   // Bond(const Bond& bd);
//   Bond bd1(BD_IN, 100);
//   Bond bd2(bd1);
//
//   EXPECT_EQ(bd1,bd2);
// }
//
// TEST(Bond, ChangeBondType){
//   // Bond& change(bondType tp);
//
//   Bond bd(BD_IN, 100);
//   bd.change(BD_OUT);
//   EXPECT_EQ(BD_OUT,bd.type());
//   // test if the qnum is inverted
// }
//
// TEST(Bond, DummyChangeBondType){
//   // Bond& dummy_change(bondType tp);
// }
//
// TEST(Bond, combine){
//   // Bond& combine(Bond bd);
//   uni10::Qnum q2(2);
//   uni10::Qnum q1(1);
//   uni10::Qnum q0(0);
//   uni10::Qnum q_1(-1);
//   uni10::Qnum q_2(-2);
//   // Create an array of Qnums for the states of a bond.
//   std::vector<uni10::Qnum> qnums;
//   qnums.push_back(q1);
//   qnums.push_back(q1);
//   qnums.push_back(q0);
//   qnums.push_back(q0);
//   qnums.push_back(q0);
//   qnums.push_back(q_1);
//
//   // Constrcut first bond
//   uni10::Bond bd(uni10::BD_IN, qnums);
//
//   // Construct another bond
//   qnums.clear();
//   qnums.push_back(q1);
//   qnums.push_back(q0);
//   qnums.push_back(q0);
//   qnums.push_back(q_1);
//   uni10::Bond bd2(uni10::BD_IN, qnums);
//   bd2.combine(bd);
//
//   //  std::cout<<"Degeneracies of bd2 after combining bd: "<<std::endl;
//   std::map<uni10::Qnum, int> degs;
//   degs = bd2.degeneracy();
//   //for(std::map<uni10::Qnum,int>::const_iterator it=degs.begin(); it!=degs.end(); ++it)
//   //	std::cout<<it->first<<": "<<it->second<<std::endl;
//   //std::cout<<std::endl;
//
//   // test bond type
//   EXPECT_EQ(BD_IN,bd2.type());
//   // test bond dimension
//   EXPECT_EQ(24, bd2.dim());
//   // test degeneracy
//   std::map<Qnum,int>::const_iterator it=degs.begin();
//   EXPECT_EQ(q_2, it->first);
//   EXPECT_EQ(1, it->second);
//
//   ++it;
//   EXPECT_EQ(q_1,it->first);
//   EXPECT_EQ(5,it->second);
//
//   ++it;
//   EXPECT_EQ(q0,it->first);
//   EXPECT_EQ(9,it->second);
//
//   ++it;
//   EXPECT_EQ(q1,it->first);
//   EXPECT_EQ(7,it->second);
//
//   ++it;
//   EXPECT_EQ(q2,it->first);
//   EXPECT_EQ(2,it->second);
// }
//
