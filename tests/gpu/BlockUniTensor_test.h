#ifndef CYTNX_TESTS_GPU_BLOCKUNITENSOR_TEST_H_
#define CYTNX_TESTS_GPU_BLOCKUNITENSOR_TEST_H_

#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

namespace cytnx {
  namespace gpu_test {

    class BlockUniTensorTest : public ::testing::Test {
     public:
      std::string data_dir = CYTNX_TEST_DATA_DIR "/common/BlockUniTensor/";

      Bond B1 = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 2});
      Bond B2 = Bond(BD_IN, {Qs(0), Qs(1)}, {3, 4});
      Bond B3 = Bond(BD_OUT, {Qs(0) >> 2, Qs(1) >> 3});
      Bond B4 = Bond(BD_OUT, {Qs(0), Qs(1)}, {1, 2});
      UniTensor BUT1 = UniTensor({B1, B2, B3, B4}).to(Device.cuda);

      Bond bd_sym_a = Bond(BD_KET, {{0, 2}, {3, 5}, {1, 6}, {4, 1}}, {4, 7, 2, 3});
      Bond bd_sym_b = Bond(BD_BRA, {{0, 2}, {3, 5}, {1, 6}, {4, 1}}, {4, 7, 2, 3});
      UniTensor BUT2 = UniTensor({bd_sym_a, bd_sym_b}).to(Device.cuda);

      Bond bd_sym_c = Bond(BD_KET, {{0, 2}, {1, 5}, {1, 6}, {0, 1}}, {4, 7, 2, 3},
                           {Symmetry::Zn(2), Symmetry::U1()});
      Bond bd_sym_d = Bond(BD_BRA, {{0, 2}, {1, 5}, {1, 6}, {0, 1}}, {4, 7, 2, 3},
                           {Symmetry::Zn(2), Symmetry::U1()});
      UniTensor BUT3 = UniTensor({bd_sym_c, bd_sym_d}).to(Device.cuda);

      Bond B1p = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {2, 1, 2});
      Bond B2p = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {4, 3, 4});
      Bond B3p = Bond(BD_IN, {Qs(-1), Qs(0), Qs(2)}, {1, 1, 1});
      Bond B4p = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {2, 1, 2});
      UniTensor BUT4 = UniTensor({B1p, B2p, B3p, B4p}).to(Device.cuda);
      UniTensor BUT4_2 = UniTensor({B1p, B2p, B3p, B4p}).to(Device.cuda);
      UniTensor BUconjT4 = UniTensor({B1p, B2p, B3p, B4p}).to(Device.cuda);
      UniTensor BUtrT4 = UniTensor({B2p, B3p}).to(Device.cuda);
      UniTensor BUTpT2 = UniTensor({B1p, B2p, B3p, B4p}).to(Device.cuda);
      UniTensor BUTsT2 = UniTensor({B1p, B2p, B3p, B4p}).to(Device.cuda);
      UniTensor BUTm9 = UniTensor({B1p, B2p, B3p, B4p}).to(Device.cuda);
      UniTensor BUTd9 = UniTensor({B1p, B2p, B3p, B4p}).to(Device.cuda);
      UniTensor BUTdT2 = UniTensor({B1p, B2p, B3p, B4p}).to(Device.cuda);

      Bond B1g = Bond(BD_IN, {Qs(1), Qs(0), Qs(0)}, {1, 1, 1});
      Bond B2g = Bond(BD_OUT, {Qs(1), Qs(0), Qs(0)}, {1, 1, 1});
      UniTensor BUT6 = UniTensor({B1g, B2g}).to(Device.cuda);

      Bond bd_sym_f = Bond(BD_KET, {{0, 2}, {1, 5}, {1, 6}, {0, 1}}, {4, 7, 2, 3},
                           {Symmetry::Zn(2), Symmetry::U1()});
      Bond bd_sym_g = Bond(BD_BRA, {{0, 2}, {1, 5}, {1, 6}, {0, 1}}, {4, 7, 2, 3},
                           {Symmetry::Zn(2), Symmetry::U1()});
      UniTensor BUT5 = UniTensor({bd_sym_f, bd_sym_g}).to(Device.cuda);

      BlockUniTensor BkUt;

      Tensor tzero345 = zeros({3, 4, 5}, Type.Double, Device.cuda);
      UniTensor utzero345 = UniTensor(zeros({3, 4, 5}, Type.Double, Device.cuda));

      Tensor t0;
      Tensor t1a;
      Tensor t1b;
      Tensor t2;

      Bond pBI = Bond(BD_IN, {Qs(0), Qs(1)}, {2, 3});
      Bond pBJ = Bond(BD_IN, {Qs(0), Qs(1)}, {4, 5});
      Bond pBK = Bond(BD_OUT, {Qs(0), Qs(1), Qs(2), Qs(3)}, {6, 7, 8, 9});

      Bond phy = Bond(BD_IN, {Qs(0), Qs(1)}, {1, 1});
      Bond aux = Bond(BD_IN, {Qs(1)}, {1});

      Bond C1B1 = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {2, 3, 4});
      Bond C1B2 = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {2, 3, 4});
      Bond C2B1 = Bond(BD_IN, {Qs(-1), Qs(0), Qs(+1), Qs(+1), Qs(0)}, {2, 2, 2, 2, 1});
      Bond C2B2 = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(+1), Qs(+1), Qs(0)}, {2, 2, 2, 2, 1});
      Bond C3B1 = Bond(BD_IN, {Qs(0), Qs(1), Qs(0), Qs(1)}, {1, 2, 3, 4});
      Bond C3B2 = Bond(BD_IN, {Qs(0), Qs(1), Qs(0), Qs(1)}, {1, 2, 3, 4});
      Bond C3B3 = Bond(BD_OUT, {Qs(0), Qs(1), Qs(2)}, {1, 2, 3});

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

      UniTensor UT_pB = UniTensor({pBI, pBJ, pBK}).to(Device.cuda);
      UniTensor UT_pB_ans = UniTensor({pBI, pBJ, pBK}).to(Device.cuda);
      UniTensor UT_contract_L1 = UniTensor({C1B1, C2B2}).to(Device.cuda);
      UniTensor UT_contract_R1 = UniTensor({C1B1, C2B2}).to(Device.cuda);
      UniTensor UT_contract_ans1 = UniTensor({C1B1, C2B2}).to(Device.cuda);
      UniTensor UT_contract_L2 = UniTensor({C2B1, C2B2}).to(Device.cuda);
      UniTensor UT_contract_R2 = UniTensor({C2B1, C2B2}).to(Device.cuda);
      UniTensor UT_contract_ans2 = UniTensor({C2B1, C2B2}).to(Device.cuda);
      UniTensor UT_contract_L3 = UniTensor({C3B1, C3B2, C3B3}).to(Device.cuda);
      UniTensor UT_contract_R3 =
        UniTensor({C3B3.redirect(), C3B1.redirect(), C3B2.redirect()}).to(Device.cuda);
      UniTensor UT_contract_ans3 =
        UniTensor({C3B1, C3B2, C3B1.redirect(), C3B2.redirect()}).to(Device.cuda);

      UniTensor UT_permute_1 = UniTensor({C3B1, C3B2, C3B3}).to(Device.cuda);
      UniTensor UT_permute_ans1 = UniTensor({C3B3, C3B1, C3B2}).to(Device.cuda);
      UniTensor UT_permute_2 = UniTensor({C3B1, C3B2.redirect()}).to(Device.cuda);
      UniTensor UT_permute_ans2 = UniTensor({C3B2.redirect(), C3B1}).to(Device.cuda);
      // UniTensor UT_permute_3 = UniTensor({C3B3.redirect(), C3B3});
      // UniTensor UT_permute_ans3 = UniTensor({C3B3, C3B3.redirect()});

      Bond Bdiag = Bond(BD_IN, {Qs(-1), Qs(1), Qs(1), Qs(-1), Qs(2)}, {3, 2, 1, 1, 5});
      UniTensor UT_diag = UniTensor({Bdiag, Bdiag.redirect()}, std::vector<std::string>({"0", "1"}),
                                    1, Type.ComplexDouble, Device.cuda, true)
                            .to(Device.cuda);
      UniTensor UT_diag_cplx =
        UniTensor({Bdiag, Bdiag.redirect()}, std::vector<std::string>({"0", "1"}), 1,
                  Type.ComplexDouble, Device.cuda, true)
          .to(Device.cuda);

     protected:
      void SetUp() override {
        BUT4 = UniTensor::Load(data_dir + "OriginalBUT.cytnx").to(Device.cuda);
        BUT4_2 = UniTensor::Load(data_dir + "OriginalBUT2.cytnx").to(Device.cuda);
        BUconjT4 = UniTensor::Load(data_dir + "BUconjT.cytnx").to(Device.cuda);
        BUtrT4 = UniTensor::Load(data_dir + "BUtrT.cytnx").to(Device.cuda);
        BUTpT2 = UniTensor::Load(data_dir + "BUTpT2.cytnx").to(Device.cuda);
        BUTsT2 = UniTensor::Load(data_dir + "BUTsT2.cytnx").to(Device.cuda);
        BUTm9 = UniTensor::Load(data_dir + "BUTm9.cytnx").to(Device.cuda);
        BUTd9 = UniTensor::Load(data_dir + "BUTd9.cytnx").to(Device.cuda);
        BUTdT2 = UniTensor::Load(data_dir + "BUTdT2.cytnx").to(Device.cuda);

        BUT6.at({0, 0}) = 1;
        BUT6.at({1, 1}) = 2;
        BUT6.at({2, 1}) = 3;
        BUT6.at({1, 2}) = 4;
        BUT6.at({2, 2}) = 5;

        t0 = Tensor::Load(data_dir + "put_block_t0.cytn").to(Device.cuda);
        t1a = Tensor::Load(data_dir + "put_block_t1a.cytn").to(Device.cuda);
        t1b = Tensor::Load(data_dir + "put_block_t1b.cytn").to(Device.cuda);
        t2 = Tensor::Load(data_dir + "put_block_t2.cytn").to(Device.cuda);

        UT_pB_ans = UniTensor::Load(data_dir + "put_block_ans.cytnx").to(Device.cuda);
        UT_contract_L1 = UniTensor::Load(data_dir + "contract_L1.cytnx").to(Device.cuda);
        UT_contract_R1 = UniTensor::Load(data_dir + "contract_R1.cytnx").to(Device.cuda);
        UT_contract_ans1 = UniTensor::Load(data_dir + "contract_ans1.cytnx").to(Device.cuda);
        UT_contract_L2 = UniTensor::Load(data_dir + "contract_L2.cytnx").to(Device.cuda);
        UT_contract_R2 = UniTensor::Load(data_dir + "contract_R2.cytnx").to(Device.cuda);
        UT_contract_ans2 = UniTensor::Load(data_dir + "contract_ans2.cytnx").to(Device.cuda);
        UT_contract_L3 = UniTensor::Load(data_dir + "contract_L3.cytnx").to(Device.cuda);
        UT_contract_R3 = UniTensor::Load(data_dir + "contract_R3.cytnx").to(Device.cuda);
        UT_contract_ans3 = UniTensor::Load(data_dir + "contract_ans3.cytnx").to(Device.cuda);

        UT_permute_1 = UniTensor::Load(data_dir + "permute_T1.cytnx").to(Device.cuda);
        UT_permute_ans1 = UniTensor::Load(data_dir + "permute_ans1.cytnx").to(Device.cuda);
        UT_permute_2 = UniTensor::Load(data_dir + "permute_T2.cytnx").to(Device.cuda);
        UT_permute_ans2 = UniTensor::Load(data_dir + "permute_ans2.cytnx").to(Device.cuda);
        // UT_permute_3 = UT_permute_3.Load(data_dir+"permute_T3.cytnx");
        // UT_permute_ans3 = UT_permute_ans3.Load(data_dir+"permute_ans3.cytnx");

        for (size_t i = 0; i < UT_diag.bonds()[0].qnums().size(); i++) {
          cytnx_uint64 deg = UT_diag.bonds()[0]._impl->_degs[i];
          UT_diag.get_block_(i).fill(static_cast<cytnx_uint64>(i + 1));
        }
        using std::complex_literals::operator""i;
        for (size_t i = 0; i < UT_diag_cplx.bonds()[0].qnums().size(); i++) {
          cytnx_uint64 deg = UT_diag_cplx.bonds()[0]._impl->_degs[i];
          UT_diag_cplx.get_block_(i).fill(cytnx_complex128{i + 1., i + 1.});
        }
      }
      void TearDown() override {}
    };

  }  // namespace gpu_test
}  // namespace cytnx
#endif  // CYTNX_TESTS_GPU_BLOCKUNITENSOR_TEST_H_
