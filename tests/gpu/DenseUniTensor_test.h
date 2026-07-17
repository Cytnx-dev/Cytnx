#ifndef CYTNX_TESTS_GPU_DENSEUNITENSOR_TEST_H_
#define CYTNX_TESTS_GPU_DENSEUNITENSOR_TEST_H_

#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"
namespace cytnx {
  namespace test {

    class DenseUniTensorTest : public ::testing::Test {
     public:
      std::string data_dir = CYTNX_TEST_DATA_DIR "/common/DenseUniTensor/";

      UniTensor utzero345;
      UniTensor utone345;
      UniTensor utar345;
      UniTensor utzero3456;
      UniTensor utone3456;
      UniTensor utar3456;
      UniTensor utarcomplex345;
      UniTensor utarcomplex3456;
      Bond phy = Bond(2, BD_IN);
      Bond aux = Bond(1, BD_IN);
      DenseUniTensor dut;
      Tensor tzero345 = zeros({3, 4, 5}, Type.ComplexDouble, Device.cuda);
      Tensor tar345 = arange(0, 3 * 4 * 5, 1, Type.ComplexDouble, Device.cuda).reshape({3, 4, 5});

      UniTensor Spf =
        UniTensor({phy, phy.redirect(), aux}, {"1", "2", "3"}, 1, Type.Float, Device.cuda, false)
          .to(Device.cuda);
      UniTensor Spd =
        UniTensor({phy, phy.redirect(), aux}, {"1", "2", "3"}, 1, Type.Double, Device.cuda, false)
          .to(Device.cuda);
      UniTensor Spcf = UniTensor({phy, phy.redirect(), aux}, {"1", "2", "3"}, 1, Type.ComplexFloat,
                                 Device.cuda, false)
                         .to(Device.cuda);
      UniTensor Spcd = UniTensor({phy, phy.redirect(), aux}, {"1", "2", "3"}, 1, Type.ComplexDouble,
                                 Device.cuda, false)
                         .to(Device.cuda);

      UniTensor ut1, ut2, contres1, contres2, contres3, dense4trtensor, densetr;
      UniTensor ut3, ut4, permu1, permu2;

     protected:
      void SetUp() override {
        using std::complex_literals::operator""i;

        utzero345 = UniTensor(zeros({3, 4, 5}, Type.ComplexDouble, Device.cuda));
        utone345 = UniTensor(ones({3, 4, 5}, Type.ComplexDouble, Device.cuda));
        utar345 =
          UniTensor(arange(0, 3 * 4 * 5, 1, Type.ComplexDouble, Device.cuda)).reshape({3, 4, 5});
        utzero3456 = UniTensor(zeros({3, 4, 5, 6}, Type.ComplexDouble, Device.cuda));
        utone3456 = UniTensor(ones({3, 4, 5, 6}, Type.ComplexDouble, Device.cuda));
        utar3456 = UniTensor(arange(0, 3 * 4 * 5 * 6, 1, Type.ComplexDouble, Device.cuda))
                     .reshape({3, 4, 5, 6});
        utarcomplex345 =
          UniTensor((1.0 + 1.0i) * arange(0, 3 * 4 * 5, 1, Type.ComplexDouble, Device.cuda))
            .reshape({3, 4, 5});
        utarcomplex3456 =
          UniTensor((1.0 + 1.0i) * arange(0, 3 * 4 * 5 * 6, 1, Type.ComplexDouble, Device.cuda))
            .reshape({3, 4, 5, 6});

        ut1 = ut1.Load(data_dir + "denseutensor1.cytnx").astype(Type.ComplexDouble).to(Device.cuda);
        ut2 = ut2.Load(data_dir + "denseutensor2.cytnx").astype(Type.ComplexDouble).to(Device.cuda);
        contres1 = contres1.Load(data_dir + "densecontres1.cytnx")
                     .astype(Type.ComplexDouble)
                     .to(Device.cuda);
        contres2 = contres2.Load(data_dir + "densecontres2.cytnx")
                     .astype(Type.ComplexDouble)
                     .to(Device.cuda);
        contres3 = contres3.Load(data_dir + "densecontres3.cytnx")
                     .astype(Type.ComplexDouble)
                     .to(Device.cuda);

        dense4trtensor = dense4trtensor.Load(data_dir + "dense4trtensor.cytnx")
                           .astype(Type.ComplexDouble)
                           .to(Device.cuda);
        densetr =
          densetr.Load(data_dir + "densetr.cytnx").astype(Type.ComplexDouble).to(Device.cuda);

        ut3 = ut3.Load(data_dir + "denseutensor3.cytnx").astype(Type.ComplexDouble).to(Device.cuda);
        ut4 = ut4.Load(data_dir + "denseutensor4.cytnx").astype(Type.ComplexDouble).to(Device.cuda);
        permu1 =
          permu1.Load(data_dir + "densepermu1.cytnx").astype(Type.ComplexDouble).to(Device.cuda);
        permu2 =
          permu2.Load(data_dir + "densepermu2.cytnx").astype(Type.ComplexDouble).to(Device.cuda);
      }
      void TearDown() override {}
    };

  }  // namespace test
}  // namespace cytnx
#endif  // CYTNX_TESTS_GPU_DENSEUNITENSOR_TEST_H_
