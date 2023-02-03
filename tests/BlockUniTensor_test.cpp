#include "BlockUniTensor_test.h"

TEST_F(BlockUniTensorTest, Trace) {
  // std::cout<<BUT4<<std::endl;
  auto tmp = BUT4.Trace(0,3);
  std::cout<<BUtrT4<<std::endl;
  std::cout<<tmp<<std::endl;
  for(size_t j=1;j<=11;j++)
      for(size_t k=1;k<=3;k++)
        if(BUtrT4.at({j-1,k-1}).exists()){
          // EXPECT_TRUE(Scalar(tmp.at({j-1,k-1})-BUtrT4.at({j-1,k-1})).abs()<1e-5);
          EXPECT_DOUBLE_EQ(double(tmp.at({j-1,k-1}).real()),double(BUtrT4.at({j-1,k-1}).real()));
          EXPECT_DOUBLE_EQ(double(tmp.at({j-1,k-1}).imag()),double(BUtrT4.at({j-1,k-1}).imag()));
        }
  // EXPECT_NO_THROW(BUT1.Trace(0,3));
  // EXPECT_THROW(BUT1.Trace(),std::logic_error);
  // EXPECT_THROW(BUT1.Trace(0,1),std::logic_error);
  // EXPECT_THROW(BUT1.Trace(-1,2),std::logic_error);
  // EXPECT_THROW(BUT1.Trace(-1,5),std::logic_error);
}

TEST_F(BlockUniTensorTest, relabels){
  BUT1 = BUT1.relabels({"a", "b", "cd", "d"});
  EXPECT_EQ(BUT1.labels()[0],"a");
  EXPECT_EQ(BUT1.labels()[1],"b");
  EXPECT_EQ(BUT1.labels()[2],"cd");
  EXPECT_EQ(BUT1.labels()[3],"d");
  BUT1 = BUT1.relabels({1,-1,2,1000});
  EXPECT_THROW(BUT1.relabels({"a","a","b","c"}), std::logic_error);
  EXPECT_THROW(BUT1.relabels({1,1,0,-1}), std::logic_error);
  EXPECT_THROW(BUT1.relabels({"a"}), std::logic_error);
  EXPECT_THROW(BUT1.relabels({1,2}), std::logic_error);
  EXPECT_THROW(BUT1.relabels({"a","b","c","d","e"}), std::logic_error);
}

TEST_F(BlockUniTensorTest, relabel){
  BUT1 = BUT1.relabel("0", "a");
  BUT1 = BUT1.relabel("1", "b");
  BUT1 = BUT1.relabel("2", "d");
  BUT1 = BUT1.relabel("3", "de");
  BUT1 = BUT1.relabel("b", "ggg");
  EXPECT_EQ(BUT1.labels()[0],"a");
  EXPECT_EQ(BUT1.labels()[1],"ggg");
  EXPECT_EQ(BUT1.labels()[2],"d");
  EXPECT_EQ(BUT1.labels()[3],"de");
  BUT1 = BUT1.relabel(0,"ccc");
  EXPECT_EQ(BUT1.labels()[0],"ccc");
  BUT1 = BUT1.relabel(0,-1);
  EXPECT_EQ(BUT1.labels()[0],"-1");
  BUT1 = BUT1.relabel(1,-199922);
  EXPECT_EQ(BUT1.labels()[1],"-199922");
  BUT1 = BUT1.relabel("-1","0");
  EXPECT_EQ(BUT1.labels()[0],"0");
  // BUT1.relabel(0,'a');
  // EXPECT_EQ(BUT1.labels()[0],"a");
  EXPECT_THROW(BUT1.relabel(5,"a"),std::logic_error);
  EXPECT_THROW(BUT1.relabel(-1,"a"),std::logic_error);
  EXPECT_THROW(BUT1.relabel(0,"a").relabel(1,"a"),std::logic_error);
  // BUT1.relabel(0,"a").relabel(1,"a");
  // EXPECT_THROW(BUT1.relabel("a","b"),std::logic_error);
  // EXPECT_THROW(BUT1.relabel(5,'a'),std::logic_error);
}

TEST_F(BlockUniTensorTest, syms){
  EXPECT_EQ(BUT1.syms(), std::vector<Symmetry>{Symmetry::U1()});
  EXPECT_EQ(BUT2.syms(), std::vector<Symmetry>({Symmetry::U1(),Symmetry::U1()}));
  EXPECT_EQ(BUT3.syms(), std::vector<Symmetry>({Symmetry::Zn(2), Symmetry::U1()}));
}

TEST_F(BlockUniTensorTest, Norm){
  // std::cout<<BUT4<<std::endl;
  // EXPECT_TRUE(Scalar(BUT4.Norm().at({0})-10.02330912178208).abs()<1e-5);
  EXPECT_DOUBLE_EQ(double(BUT4.Norm().at({0}).real()),10.07992623704349);
}

TEST_F(BlockUniTensorTest, Conj){
  auto tmp = BUT4.Conj();
  for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
      for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
        if(BUconjT4.at({i-1,j-1,k-1,l-1}).exists()){
          EXPECT_TRUE(Scalar(tmp.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
        }
  BUT4.Conj_();
  for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
      for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
        if(BUT4.at({i-1,j-1,k-1,l-1}).exists()){
          EXPECT_TRUE(Scalar(BUT4.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
        }
}

TEST_F(BlockUniTensorTest, Transpose){
  auto tmp = BUT1.Transpose();
  EXPECT_EQ(tmp.bonds()[0].type(), BD_OUT);
  EXPECT_EQ(tmp.bonds()[1].type(), BD_OUT);
  EXPECT_EQ(tmp.bonds()[2].type(), BD_IN);
  EXPECT_EQ(tmp.bonds()[3].type(), BD_IN);

  tmp = BUT5.Transpose();
  EXPECT_EQ(tmp.bonds()[0].type(), BD_BRA);
  EXPECT_EQ(tmp.bonds()[1].type(), BD_KET);
  EXPECT_EQ(tmp.bonds()[0].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6}, {0,1}}));
  EXPECT_EQ(tmp.bonds()[1].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6}, {0,1}}));

  BUT1.Transpose_();
  EXPECT_EQ(BUT1.bonds()[0].type(), BD_OUT);
  EXPECT_EQ(BUT1.bonds()[1].type(), BD_OUT);
  EXPECT_EQ(BUT1.bonds()[2].type(), BD_IN);
  EXPECT_EQ(BUT1.bonds()[3].type(), BD_IN);

  BUT5.Transpose_();
  EXPECT_EQ(BUT5.bonds()[0].type(), BD_BRA);
  EXPECT_EQ(BUT5.bonds()[1].type(), BD_KET);
  EXPECT_EQ(BUT5.bonds()[0].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6}, {0,1}}));
  EXPECT_EQ(BUT5.bonds()[1].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6}, {0,1}}));
}

TEST_F(BlockUniTensorTest, Dagger){
  auto tmp = BUT1.Dagger();
  EXPECT_EQ(tmp.bonds()[0].type(), BD_OUT);
  EXPECT_EQ(tmp.bonds()[1].type(), BD_OUT);
  EXPECT_EQ(tmp.bonds()[2].type(), BD_IN);
  EXPECT_EQ(tmp.bonds()[3].type(), BD_IN);

  tmp = BUT5.Dagger();
  EXPECT_EQ(tmp.bonds()[0].type(), BD_BRA);
  EXPECT_EQ(tmp.bonds()[1].type(), BD_KET);
  EXPECT_EQ(tmp.bonds()[0].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6}, {0,1}}));
  EXPECT_EQ(tmp.bonds()[1].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6}, {0,1}}));

  BUT1.Dagger_();
  EXPECT_EQ(BUT1.bonds()[0].type(), BD_OUT);
  EXPECT_EQ(BUT1.bonds()[1].type(), BD_OUT);
  EXPECT_EQ(BUT1.bonds()[2].type(), BD_IN);
  EXPECT_EQ(BUT1.bonds()[3].type(), BD_IN);

  BUT5.Dagger_();
  EXPECT_EQ(BUT5.bonds()[0].type(), BD_BRA);
  EXPECT_EQ(BUT5.bonds()[1].type(), BD_KET);
  EXPECT_EQ(BUT5.bonds()[0].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6}, {0,1}}));
  EXPECT_EQ(BUT5.bonds()[1].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6}, {0,1}}));

  tmp = BUT4.Dagger();
  for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
      for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
        if(BUconjT4.at({i-1,j-1,k-1,l-1}).exists()){
          // EXPECT_TRUE(Scalar(tmp.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
          EXPECT_DOUBLE_EQ(double(tmp.at({i-1,j-1,k-1,l-1}).real()),double(BUconjT4.at({i-1,j-1,k-1,l-1}).real()));
          EXPECT_DOUBLE_EQ(double(tmp.at({i-1,j-1,k-1,l-1}).imag()),double(BUconjT4.at({i-1,j-1,k-1,l-1}).imag()));
        }
  
  BUT4.Dagger_();
  for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
      for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
        if(BUT4.at({i-1,j-1,k-1,l-1}).exists()){
          // EXPECT_TRUE(Scalar(BUT4.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
          EXPECT_DOUBLE_EQ(double(BUT4.at({i-1,j-1,k-1,l-1}).real()),double(BUconjT4.at({i-1,j-1,k-1,l-1}).real()));
          EXPECT_DOUBLE_EQ(double(BUT4.at({i-1,j-1,k-1,l-1}).imag()),double(BUconjT4.at({i-1,j-1,k-1,l-1}).imag()));
        }
}

TEST_F(BlockUniTensorTest, truncate){
  auto tmp = BUT5.truncate(0,1);
  EXPECT_EQ(tmp.bonds()[0].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,6}, {0,1}}));
  EXPECT_EQ(tmp.bonds()[1].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6}, {0,1}}));
  tmp = BUT5.truncate(1,0);
  EXPECT_EQ(tmp.bonds()[0].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6}, {0,1}}));
  EXPECT_EQ(tmp.bonds()[1].qnums(),std::vector<std::vector<cytnx_int64>>({{1,5}, {1,6}, {0,1}}));
  BUT5.truncate_(1,3);
  EXPECT_EQ(BUT5.bonds()[0].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6}, {0,1}}));
  EXPECT_EQ(BUT5.bonds()[1].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6}}));
  EXPECT_THROW(BUT5.truncate(-1,1), std::logic_error);
  EXPECT_THROW(BUT5.truncate(0,-1), std::logic_error);
  EXPECT_THROW(BUT5.truncate(0,4), std::logic_error);
  EXPECT_THROW(BUT5.truncate(2,0), std::logic_error);
  EXPECT_THROW(BUT5.truncate_(-1,1), std::logic_error);
  EXPECT_THROW(BUT5.truncate_(0,-1), std::logic_error);
  EXPECT_THROW(BUT5.truncate_(0,4), std::logic_error);
  EXPECT_THROW(BUT5.truncate_(2,0), std::logic_error);
}

TEST_F(BlockUniTensorTest, elem_exist){
  for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
      for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
          if(BUT4.elem_exists({i-1,j-1,k-1,l-1})){
            cytnx_int64 _a;
            std::vector<cytnx_uint64> _b;
            ((BlockUniTensor*)BUT4._impl.get())->_fx_locate_elem(_a,_b,{i-1,j-1,k-1,l-1});
            std::vector<cytnx_uint64> qind = BUT4.get_qindices(_a);
            EXPECT_EQ(BUT4.bonds()[0].qnums()[qind[0]][0]-
            BUT4.bonds()[1].qnums()[qind[1]][0]+
            BUT4.bonds()[2].qnums()[qind[2]][0]-
            BUT4.bonds()[3].qnums()[qind[3]][0],0);
          }
  EXPECT_THROW(BUT4.elem_exists({100,0,0,0}),std::logic_error);
  EXPECT_THROW(BUT4.elem_exists({1,0,0,0,0}),std::logic_error);
  EXPECT_THROW(BUT4.elem_exists({0,0,0}),std::logic_error);
  EXPECT_THROW(BUT4.elem_exists({}),std::logic_error);
}

