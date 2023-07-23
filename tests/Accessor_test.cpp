#include "Accessor_test.h"

// Test Accessor::type() method
TEST_F(AccessorTest, Type) {
  EXPECT_EQ(single.type(), cytnx::Accessor::Singl);
  EXPECT_EQ(all.type(), cytnx::Accessor::All);
  EXPECT_EQ(range.type(), cytnx::Accessor::Range);
  EXPECT_EQ(tilend.type(), cytnx::Accessor::Tilend);
  EXPECT_EQ(step.type(), cytnx::Accessor::Step);
  EXPECT_EQ(list.type(), cytnx::Accessor::list);
}

// Test Accessor::all() method
TEST_F(AccessorTest, All) { EXPECT_EQ(all.type(), cytnx::Accessor::All); }

// Test Accessor::range() method
TEST_F(AccessorTest, Range) {
  EXPECT_EQ(range.type(), cytnx::Accessor::Range);
  EXPECT_EQ(range._min, 1);
  EXPECT_EQ(range._max, 4);
  EXPECT_EQ(range._step, 2);
}

// Test Accessor::tilend() method
TEST_F(AccessorTest, Tilend) {
  EXPECT_EQ(tilend.type(), cytnx::Accessor::Tilend);
  EXPECT_EQ(tilend._min, 2);
  EXPECT_EQ(tilend._step, 1);
}

// Test Accessor::step() method
TEST_F(AccessorTest, Step) {
  EXPECT_EQ(step.type(), cytnx::Accessor::Step);
  EXPECT_EQ(step._step, 3);
}

// Test Accessor::constructor with list
TEST_F(AccessorTest, ConstructorWithList) {
  EXPECT_EQ(list.type(), cytnx::Accessor::list);
  EXPECT_EQ(list.idx_list.size(), 3);
  EXPECT_EQ(list.idx_list[0], 0);
  EXPECT_EQ(list.idx_list[1], 2);
  EXPECT_EQ(list.idx_list[2], 3);
}

TEST_F(AccessorTest, ListConstructor) {
  std::vector<int> idx_list = {0, 2, 4};
  cytnx::Accessor A(idx_list);
  EXPECT_EQ(A.type(), cytnx::Accessor::list);
  for (size_t i = 0; i < idx_list.size(); i++) {
    EXPECT_EQ(A.idx_list[i], idx_list[i]);
  }
  // EXPECT_EQ(A.idx_list, idx_list);

  std::initializer_list<int> init_list = {1, 3, 5};
  cytnx::Accessor B(init_list);
  EXPECT_EQ(B.type(), cytnx::Accessor::list);
  std::vector<int> expected_idx_list(init_list.begin(), init_list.end());
  for (size_t i = 0; i < expected_idx_list.size(); i++) {
    EXPECT_EQ(B.idx_list[i], expected_idx_list[i]);
  }
}

TEST_F(AccessorTest, AllConstructor) {
  cytnx::Accessor A(":");
  EXPECT_EQ(A.type(), cytnx::Accessor::All);

  cytnx::Accessor B = cytnx::Accessor::all();
  EXPECT_EQ(B.type(), cytnx::Accessor::All);
}

TEST_F(AccessorTest, RangeConstructor) {
  cytnx::Accessor A(1, 5, 2);
  EXPECT_EQ(A.type(), cytnx::Accessor::Range);
  EXPECT_EQ(A._min, 1);
  EXPECT_EQ(A._max, 5);
  EXPECT_EQ(A._step, 2);

  cytnx::Accessor B = cytnx::Accessor::range(3, 9, 3);
  EXPECT_EQ(B.type(), cytnx::Accessor::Range);
  EXPECT_EQ(B._min, 3);
  EXPECT_EQ(B._max, 9);
  EXPECT_EQ(B._step, 3);
}

TEST_F(AccessorTest, TilendConstructor) {
  cytnx::Accessor A = cytnx::Accessor::tilend(2, 3);
  EXPECT_EQ(A.type(), cytnx::Accessor::Tilend);
  EXPECT_EQ(A._min, 2);
  EXPECT_EQ(A._step, 3);

  EXPECT_THROW(cytnx::Accessor::tilend(2, 0), std::logic_error);
}

TEST_F(AccessorTest, StepConstructor) {
  cytnx::Accessor A = cytnx::Accessor::step(3);
  EXPECT_EQ(A.type(), cytnx::Accessor::Step);
  EXPECT_EQ(A._step, 3);

  EXPECT_THROW(cytnx::Accessor::step(0), std::logic_error);
}

TEST_F(AccessorTest, CopyConstructor) {
  cytnx::Accessor A(2);
  cytnx::Accessor B = A;
  EXPECT_EQ(A.type(), B.type());
  EXPECT_EQ(A.loc, B.loc);
  EXPECT_EQ(A.idx_list, B.idx_list);
  EXPECT_EQ(A._min, B._min);
  EXPECT_EQ(A._max, B._max);
  EXPECT_EQ(A._step, B._step);
}

TEST_F(AccessorTest, CopyAssignment) {
  cytnx::Accessor A(2);
  cytnx::Accessor B;
  B = A;
  EXPECT_EQ(A.type(), B.type());
  EXPECT_EQ(A.loc, B.loc);
  EXPECT_EQ(A.idx_list, B.idx_list);
  EXPECT_EQ(A._min, B._min);
  EXPECT_EQ(A._max, B._max);
  EXPECT_EQ(A._step, B._step);
  EXPECT_EQ(A.type(), B.type());
}

TEST_F(AccessorTest, SingleConstructor) {
  cytnx::Accessor acc1(1);
  ASSERT_EQ(acc1.type(), cytnx::Accessor::Singl);
  ASSERT_EQ(acc1._min, 0);
  ASSERT_EQ(acc1._max, 0);
  ASSERT_EQ(acc1._step, 0);
  ASSERT_EQ(acc1.loc, 1);

  cytnx::Accessor acc2(-1);
  ASSERT_EQ(acc2.type(), cytnx::Accessor::Singl);
  ASSERT_EQ(acc2._min, 0);
  ASSERT_EQ(acc2._max, 0);
  ASSERT_EQ(acc2._step, 0);
  ASSERT_EQ(acc2.loc, -1);
}

TEST_F(AccessorTest, AllGenerator) {
  cytnx::Accessor acc = cytnx::Accessor::all();
  ASSERT_EQ(acc.type(), cytnx::Accessor::All);
}
