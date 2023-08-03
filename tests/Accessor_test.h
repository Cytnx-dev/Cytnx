#include "cytnx.hpp"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// Test fixture for Accessor class
class AccessorTest : public ::testing::Test {
 protected:
  // Declare member variables that are used in the tests
  cytnx::Accessor single, all, range, tilend, step, list;

  // Set up the test fixture
  void SetUp() override {
    single = cytnx::Accessor(5);
    all = cytnx::Accessor::all();
    range = cytnx::Accessor::range(1, 4, 2);
    tilend = cytnx::Accessor::tilend(2, 1);
    step = cytnx::Accessor::step(3);
    list = cytnx::Accessor({0, 2, 3});
  }
};
