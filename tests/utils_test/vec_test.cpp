#include "vec_test.h"

TEST_F(VecTest, vec_concatenate) {
	ui64v.push_back(1);
	EXPECT_EQ(ui64v.at(0),1);
}
