#ifndef _H_DENSEUNITENSOR_TEST
#define _H_DENSEUNITENSOR_TEST

#include "cytnx.hpp"
#include <gtest/gtest.h>
#include "test_tools.h"

using namespace cytnx;
using namespace std;
using namespace TestTools;
class DenseUniTensorTest : public ::testing::Test {
 public:
  std::string data_dir = "../../tests/test_data_base/common/DenseUniTensor/";

  UniTensor ut_uninit;
  UniTensor utzero345;
  UniTensor utone345;
  UniTensor utar345;
  UniTensor utzero3456;
  UniTensor utone3456;
  UniTensor utar3456;
  UniTensor utarcomplex345;
  UniTensor utarcomplex3456;
  UniTensor ut_complex_diag;
  Bond phy = Bond(2, BD_IN);
  Bond aux = Bond(1, BD_IN);
  DenseUniTensor dut;
  Tensor tzero345 = zeros({3, 4, 5}).astype(Type.ComplexDouble);
  Tensor tar345 = arange({3 * 4 * 5}).reshape({3, 4, 5}).astype(Type.ComplexDouble);

  UniTensor Spf =
    UniTensor({phy, phy.redirect(), aux}, {"1", "2", "3"}, 1, Type.Float, Device.cpu, false);
  UniTensor Spd =
    UniTensor({phy, phy.redirect(), aux}, {"1", "2", "3"}, 1, Type.Double, Device.cpu, false);
  UniTensor Spcf =
    UniTensor({phy, phy.redirect(), aux}, {"1", "2", "3"}, 1, Type.ComplexFloat, Device.cpu, false);
  UniTensor Spcd = UniTensor({phy, phy.redirect(), aux}, {"1", "2", "3"}, 1, Type.ComplexDouble,
                             Device.cpu, false);

  UniTensor ut1, ut2, contres1, contres2, contres3, dense4trtensor, densetr;
  UniTensor ut3, ut4, permu1, permu2;

 protected:
  void SetUp() override {
    utzero345 = UniTensor(zeros(3 * 4 * 5)).reshape({3, 4, 5}).astype(Type.ComplexDouble);
    utone345 = UniTensor(ones(3 * 4 * 5)).reshape({3, 4, 5}).astype(Type.ComplexDouble);
    utar345 = UniTensor(arange(3 * 4 * 5)).reshape({3, 4, 5}).astype(Type.ComplexDouble);
    utzero3456 = UniTensor(zeros(3 * 4 * 5 * 6)).reshape({3, 4, 5, 6}).astype(Type.ComplexDouble);
    utone3456 = UniTensor(ones(3 * 4 * 5 * 6)).reshape({3, 4, 5, 6}).astype(Type.ComplexDouble);
    utar3456 = UniTensor(arange(3 * 4 * 5 * 6)).reshape({3, 4, 5, 6}).astype(Type.ComplexDouble);
    utarcomplex345 = UniTensor(arange(3 * 4 * 5)).astype(Type.ComplexDouble);
    for (size_t i = 0; i < 3 * 4 * 5; i++) utarcomplex345.at({i}) = cytnx_complex128(i, i);
    utarcomplex345 = utarcomplex345.reshape({3, 4, 5}).astype(Type.ComplexDouble);
    utarcomplex3456 = UniTensor(arange(3 * 4 * 5 * 6)).astype(Type.ComplexDouble);
    for (size_t i = 0; i < 3 * 4 * 5 * 6; i++) utarcomplex3456.at({i}) = cytnx_complex128(i, i);
    utarcomplex3456 = utarcomplex3456.reshape({3, 4, 5, 6}).astype(Type.ComplexDouble);
    ut_complex_diag =
      UniTensor({phy, phy.redirect()}, {"1", "2"}, 1, Type.ComplexDouble, Device.cpu, true);

    ut1 = ut1.Load(data_dir + "denseutensor1.cytnx").astype(Type.ComplexDouble);
    ut2 = ut2.Load(data_dir + "denseutensor2.cytnx").astype(Type.ComplexDouble);
    contres1 = contres1.Load(data_dir + "densecontres1.cytnx").astype(Type.ComplexDouble);
    contres2 = contres2.Load(data_dir + "densecontres2.cytnx").astype(Type.ComplexDouble);
    contres3 = contres3.Load(data_dir + "densecontres3.cytnx").astype(Type.ComplexDouble);

    dense4trtensor =
      dense4trtensor.Load(data_dir + "dense4trtensor.cytnx").astype(Type.ComplexDouble);
    densetr = densetr.Load(data_dir + "densetr.cytnx").astype(Type.ComplexDouble);

    ut3 = ut3.Load(data_dir + "denseutensor3.cytnx").astype(Type.ComplexDouble);
    ut4 = ut4.Load(data_dir + "denseutensor4.cytnx").astype(Type.ComplexDouble);
    permu1 = permu1.Load(data_dir + "densepermu1.cytnx").astype(Type.ComplexDouble);
    permu2 = permu2.Load(data_dir + "densepermu2.cytnx").astype(Type.ComplexDouble);
  }
  void TearDown() override {}
};

#endif
