#ifndef _H_Network_test
#define _H_Network_test

#include "cytnx.hpp"
#include <gtest/gtest.h>

class NetworkTest : public ::testing::Test {
 public:
  cytnx::Network NEmpty;
  cytnx::Network NetFromFile = cytnx::Network("testNet.net");

 protected:
  void SetUp() override {}
  void TearDown() override {
    NEmpty = cytnx::Network();
    NetFromFile = cytnx::Network("testNet.net");
  }
};

#endif
