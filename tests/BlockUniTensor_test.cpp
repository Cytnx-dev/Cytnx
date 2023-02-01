#include "BlockUniTensor_test.h"

TEST_F(BlockUniTensorTest, Trace) {
  auto tmp = BUT4.Trace(0,3);
  for(size_t j=1;j<=11;j++)
      for(size_t k=1;k<=3;k++)
        if(BUtrT4.at({j-1,k-1}).exists()){
          EXPECT_TRUE(Scalar(tmp.at({j-1,k-1})-BUtrT4.at({j-1,k-1})).abs()<1e-5);
        }
  EXPECT_NO_THROW(BUT1.Trace("0","3"));
  EXPECT_THROW(BUT1.Trace(),std::logic_error);
  EXPECT_THROW(BUT1.Trace("0","1"),std::logic_error);
  EXPECT_THROW(BUT1.Trace(-1,2),std::logic_error);
  EXPECT_THROW(BUT1.Trace(-1,5),std::logic_error);

}
