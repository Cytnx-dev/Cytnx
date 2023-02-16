#ifndef _H_DENSEUNITENSOR_TEST
#define _H_DENSEUNITENSOR_TEST

#include "cytnx.hpp"
#include <gtest/gtest.h>

using namespace cytnx;
using namespace std;
class DenseUniTensorTest : public ::testing::Test {
 public:
  UniTensor utzero345;
  UniTensor utone345;
  UniTensor utar345;
  UniTensor utzero3456;
  UniTensor utone3456;
  UniTensor utar3456;
  UniTensor utarcomplex345;
  UniTensor utarcomplex3456;
  Bond phy = Bond(2,BD_REG);
  Bond aux = Bond(1,BD_REG);
  DenseUniTensor dut;
  Tensor tzero345 = zeros({3, 4, 5}).astype(Type.ComplexDouble);

  UniTensor Spf = UniTensor({phy,phy.redirect(),aux},{1,2,3},1,Type.Float,Device.cpu,false).astype(Type.ComplexDouble);
  UniTensor Spd = UniTensor({phy,phy.redirect(),aux},{1,2,3},1,Type.Double,Device.cpu,false).astype(Type.ComplexDouble);
  UniTensor Spcf = UniTensor({phy,phy.redirect(),aux},{1,2,3},1,Type.ComplexFloat,Device.cpu,false).astype(Type.ComplexDouble);
  UniTensor Spcd = UniTensor({phy,phy.redirect(),aux},{1,2,3},1,Type.ComplexDouble,Device.cpu,false).astype(Type.ComplexDouble);

  UniTensor ut1,ut2,contres1,contres2,contres3;
 protected:
  void SetUp() override {
    utzero345 = UniTensor(zeros(3 * 4 * 5)).reshape({3, 4, 5}).astype(Type.ComplexDouble);
    utone345 = UniTensor(ones(3 * 4 * 5)).reshape({3, 4, 5}).astype(Type.ComplexDouble);
    utar345 = UniTensor(arange(3*4*5)).reshape({3, 4, 5}).astype(Type.ComplexDouble);
    utzero3456 = UniTensor(zeros(3 * 4 * 5 * 6)).reshape({3, 4, 5, 6}).astype(Type.ComplexDouble);
    utone3456 = UniTensor(ones(3 * 4 * 5 * 6)).reshape({3, 4, 5, 6}).astype(Type.ComplexDouble);
    utar3456 = UniTensor(arange(3*4*5*6)).reshape({3, 4, 5, 6}).astype(Type.ComplexDouble);
    utarcomplex345 = UniTensor(arange(3*4*5)).astype(Type.ComplexDouble);
    for(size_t i=0;i<3*4*5;i++) utarcomplex345.at({i}) = cytnx_complex128(i,i);
    utarcomplex345 = utarcomplex345.reshape({3, 4, 5}).astype(Type.ComplexDouble);
    utarcomplex3456 = UniTensor(arange(3*4*5*6)).astype(Type.ComplexDouble);
    for(size_t i=0;i<3*4*5*6;i++) utarcomplex3456.at({i}) = cytnx_complex128(i,i);
    utarcomplex3456 = utarcomplex3456.reshape({3, 4, 5, 6}).astype(Type.ComplexDouble);

    ut1 = ut1.Load("denseutensor1.cytnx").astype(Type.ComplexDouble);
    ut2 = ut2.Load("denseutensor2.cytnx").astype(Type.ComplexDouble);
    contres1 = contres1.Load("densecontres1.cytnx").astype(Type.ComplexDouble);
    contres2 = contres2.Load("densecontres2.cytnx").astype(Type.ComplexDouble);
    contres3 = contres3.Load("densecontres3.cytnx").astype(Type.ComplexDouble);
  }
  void TearDown() override {}
};

#endif