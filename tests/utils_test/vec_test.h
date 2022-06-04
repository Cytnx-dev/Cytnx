#pragma once
#include "cytnx.hpp"
#include <gtest/gtest.h>

class VecTest : public ::testing::Test {
public:
	std::vector<cytnx::cytnx_uint64> ui64v;
protected:
	void SetUp() override {
		
	}
	
	void TearDown() override {
		ui64v.clear();
	}
};
