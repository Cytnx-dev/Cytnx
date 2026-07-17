#ifndef CYTNX_TESTS_DENSEUNITENSOR_TEST_H_
#define CYTNX_TESTS_DENSEUNITENSOR_TEST_H_

#include <cstdio>
#include <filesystem>

#include <gtest/gtest.h>

#include "cytnx.hpp"
#include "test_tools.h"
namespace cytnx {
  namespace test {

    class DenseUniTensorTest : public ::testing::Test {
     public:
      std::string data_dir = CYTNX_TEST_DATA_DIR "/common/DenseUniTensor/";
      const std::string temp_file_path = std::string(std::tmpnam(nullptr)) + ".cytnx";

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
      Bond bond4 = Bond(4, BD_IN);
      DenseUniTensor dut;
      Tensor tzero345 = zeros({3, 4, 5}, Type.ComplexDouble);
      Tensor tar345 = arange(0, 3 * 4 * 5, 1, Type.ComplexDouble).reshape({3, 4, 5});

      UniTensor Spf =
        UniTensor({phy, phy.redirect(), aux}, {"1", "2", "3"}, 1, Type.Float, Device.cpu, false);
      UniTensor Spd =
        UniTensor({phy, phy.redirect(), aux}, {"1", "2", "3"}, 1, Type.Double, Device.cpu, false);
      UniTensor Spcf = UniTensor({phy, phy.redirect(), aux}, {"1", "2", "3"}, 1, Type.ComplexFloat,
                                 Device.cpu, false);
      UniTensor Spcd = UniTensor({phy, phy.redirect(), aux}, {"1", "2", "3"}, 1, Type.ComplexDouble,
                                 Device.cpu, false);

      UniTensor ut1, ut2, contres1, contres2, contres3, dense4trtensor, densetr;
      UniTensor ut3, ut4, permu1, permu2;

      ~DenseUniTensorTest() {
        try {
          std::filesystem::remove(temp_file_path);
        } catch (const std::filesystem::filesystem_error& error) {
          // Do nothing. The system cleans the temp files periodically.
        }
      }

     protected:
      void SetUp() override {
        using std::complex_literals::operator""i;

        utzero345 = UniTensor(zeros({3, 4, 5}, Type.ComplexDouble));
        utone345 = UniTensor(ones({3, 4, 5}, Type.ComplexDouble));
        utar345 = UniTensor(arange(0, 3 * 4 * 5, 1, Type.ComplexDouble)).reshape({3, 4, 5});
        utar345.set_labels({"a", "b", "c"}).set_name("utar345").set_rowrank_(2);
        utzero3456 = UniTensor(zeros({3, 4, 5, 6}, Type.ComplexDouble));
        utone3456 = UniTensor(ones({3, 4, 5, 6}, Type.ComplexDouble));
        utar3456 = UniTensor(arange(0, 3 * 4 * 5 * 6, 1, Type.ComplexDouble)).reshape({3, 4, 5, 6});
        utarcomplex345 =
          UniTensor((1.0 + 1.0i) * arange(0, 3 * 4 * 5, 1, Type.ComplexDouble)).reshape({3, 4, 5});
        utarcomplex3456 = UniTensor((1.0 + 1.0i) * arange(0, 3 * 4 * 5 * 6, 1, Type.ComplexDouble))
                            .reshape({3, 4, 5, 6});
        ut_complex_diag = UniTensor({bond4, bond4.redirect()}, {"row", "col"}, 1,
                                    Type.ComplexDouble, Device.cpu, true);
        ut_complex_diag.put_block(arange(0, 4, 1, Type.ComplexDouble));
        ut_complex_diag.set_name("ut_complex_diag");

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

  }  // namespace test
}  // namespace cytnx
#endif  // CYTNX_TESTS_DENSEUNITENSOR_TEST_H_
