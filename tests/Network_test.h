#ifndef _H_Network_test
#define _H_Network_test

#include "cytnx.hpp"
#include <gtest/gtest.h>

class NetworkTest : public ::testing::Test {
 public:
  Network NEmpty;
  Network NetFromFile = Network("testNet.net");

 protected:
  void SetUp() override {}
  void TearDown() override {
    NEmpty = Network();
    NetFromFile = Network("testNet.net");
  }
};

#endif
