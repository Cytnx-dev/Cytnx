#include "Accessor_test.h"

#include <initializer_list>
#include <stdexcept>
#include <vector>

// Test Accessor::type() method

namespace cytnx {
  namespace test {
    namespace {

      TEST_F(AccessorTest, Type) {
        EXPECT_EQ(single.type(), Accessor::Singl);
        EXPECT_EQ(all.type(), Accessor::All);
        EXPECT_EQ(range.type(), Accessor::Range);
        EXPECT_EQ(tilend.type(), Accessor::Tilend);
        EXPECT_EQ(step.type(), Accessor::Step);
        EXPECT_EQ(list.type(), Accessor::list);
      }

      // Test Accessor::all() method
      TEST_F(AccessorTest, All) { EXPECT_EQ(all.type(), Accessor::All); }

      // Test Accessor::range() method
      TEST_F(AccessorTest, Range) {
        EXPECT_EQ(range.type(), Accessor::Range);
        EXPECT_EQ(range._min, 1);
        EXPECT_EQ(range._max, 4);
        EXPECT_EQ(range._step, 2);
      }

      // Test Accessor::tilend() method
      TEST_F(AccessorTest, Tilend) {
        EXPECT_EQ(tilend.type(), Accessor::Tilend);
        EXPECT_EQ(tilend._min, 2);
        EXPECT_EQ(tilend._step, 1);
      }

      // Test Accessor::step() method
      TEST_F(AccessorTest, Step) {
        EXPECT_EQ(step.type(), Accessor::Step);
        EXPECT_EQ(step._step, 3);
      }

      // Test Accessor::constructor with list
      TEST_F(AccessorTest, ConstructorWithList) {
        EXPECT_EQ(list.type(), Accessor::list);
        EXPECT_EQ(list.idx_list.size(), 3);
        EXPECT_EQ(list.idx_list[0], 0);
        EXPECT_EQ(list.idx_list[1], 2);
        EXPECT_EQ(list.idx_list[2], 3);
      }

      TEST_F(AccessorTest, ListConstructor) {
        std::vector<int> idx_list = {0, 2, 4};
        Accessor A(idx_list);
        EXPECT_EQ(A.type(), Accessor::list);
        for (size_t i = 0; i < idx_list.size(); i++) {
          EXPECT_EQ(A.idx_list[i], idx_list[i]);
        }
        // EXPECT_EQ(A.idx_list, idx_list);

        std::initializer_list<int> init_list = {1, 3, 5};
        Accessor B(init_list);
        EXPECT_EQ(B.type(), Accessor::list);
        std::vector<int> expected_idx_list(init_list.begin(), init_list.end());
        for (size_t i = 0; i < expected_idx_list.size(); i++) {
          EXPECT_EQ(B.idx_list[i], expected_idx_list[i]);
        }
      }

      TEST_F(AccessorTest, AllConstructor) {
        Accessor A(":");
        EXPECT_EQ(A.type(), Accessor::All);

        Accessor B = Accessor::all();
        EXPECT_EQ(B.type(), Accessor::All);
      }

      TEST_F(AccessorTest, RangeConstructor) {
        Accessor A(1, 5, 2);
        EXPECT_EQ(A.type(), Accessor::Range);
        EXPECT_EQ(A._min, 1);
        EXPECT_EQ(A._max, 5);
        EXPECT_EQ(A._step, 2);

        Accessor B = Accessor::range(3, 9, 3);
        EXPECT_EQ(B.type(), Accessor::Range);
        EXPECT_EQ(B._min, 3);
        EXPECT_EQ(B._max, 9);
        EXPECT_EQ(B._step, 3);
      }

      TEST_F(AccessorTest, TilendConstructor) {
        Accessor A = Accessor::tilend(2, 3);
        EXPECT_EQ(A.type(), Accessor::Tilend);
        EXPECT_EQ(A._min, 2);
        EXPECT_EQ(A._step, 3);

        EXPECT_THROW(Accessor::tilend(2, 0), std::logic_error);
      }

      TEST_F(AccessorTest, StepConstructor) {
        Accessor A = Accessor::step(3);
        EXPECT_EQ(A.type(), Accessor::Step);
        EXPECT_EQ(A._step, 3);

        EXPECT_THROW(Accessor::step(0), std::logic_error);
      }

      TEST_F(AccessorTest, CopyConstructor) {
        Accessor A(2);
        Accessor B = A;
        EXPECT_EQ(A.type(), B.type());
        EXPECT_EQ(A.loc, B.loc);
        EXPECT_EQ(A.idx_list, B.idx_list);
        EXPECT_EQ(A._min, B._min);
        EXPECT_EQ(A._max, B._max);
        EXPECT_EQ(A._step, B._step);
      }

      TEST_F(AccessorTest, CopyAssignment) {
        Accessor A(2);
        Accessor B;
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
        Accessor acc1(1);
        ASSERT_EQ(acc1.type(), Accessor::Singl);
        ASSERT_EQ(acc1._min, 0);
        ASSERT_EQ(acc1._max, 0);
        ASSERT_EQ(acc1._step, 0);
        ASSERT_EQ(acc1.loc, 1);

        Accessor acc2(-1);
        ASSERT_EQ(acc2.type(), Accessor::Singl);
        ASSERT_EQ(acc2._min, 0);
        ASSERT_EQ(acc2._max, 0);
        ASSERT_EQ(acc2._step, 0);
        ASSERT_EQ(acc2.loc, -1);
      }

      TEST_F(AccessorTest, AllGenerator) {
        Accessor acc = Accessor::all();
        ASSERT_EQ(acc.type(), Accessor::All);
      }

      TEST_F(AccessorTest, Equality) {
        EXPECT_TRUE(single == Accessor(5));
        EXPECT_TRUE(all == Accessor::all());
        EXPECT_TRUE(range == Accessor::range(1, 4, 2));
        EXPECT_TRUE(tilend == Accessor::tilend(2, 1));
        EXPECT_TRUE(step == Accessor::step(3));
        EXPECT_TRUE(list == Accessor({0, 2, 3}));
        EXPECT_FALSE(single == Accessor(4));
        EXPECT_FALSE(range == Accessor::range(1, 6, 2));
        EXPECT_FALSE(tilend == Accessor::tilend(2, 2));
        EXPECT_FALSE(step == Accessor::step(4));
        EXPECT_FALSE(list == Accessor({0, 1, 3}));
        EXPECT_FALSE(range == all);
        EXPECT_FALSE(tilend == single);
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
