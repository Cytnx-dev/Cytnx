#include "Network_test.h"

#include <algorithm>
#include <complex>
#include <functional>
#include <map>
#include <string>
#include <vector>

// TEST_F(NetworkTest, GpuNetworkStringLbl) {
//   auto Hi = Network();
//   EXPECT_NO_THROW(Hi.FromString({"A: a,e", "B: a,c_,d_,h", "C: e,f_,g_,h", "TOUT:
//   c_,d_;f_,g_"})); EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {ut1, ut2, ut3}));
//   EXPECT_NO_THROW(Hi.Launch(false));
//   EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3}));
//   EXPECT_NO_THROW(Hi.Launch(false));
// }

// TEST_F(NetworkTest, GpuNetworkIntegerLbl) {
//   auto Hi = Network();
//   EXPECT_NO_THROW(Hi.FromString({"A: 1,2", "B: 1,3,4,7", "C: 2,5,6,7", "TOUT: 3,4;5,6"}));
//   EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {ut1, ut2, ut3}));
//   EXPECT_NO_THROW(Hi.Launch(false));
//   EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3}));
//   EXPECT_NO_THROW(Hi.Launch(false));
// }

namespace cytnx {
  namespace gpu_test {
    namespace {

      TEST_F(NetworkTest, GpuNetworkDenseFromString) {
        auto net = Network();
        net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
      }

      TEST_F(NetworkTest, GpuNetworkDenseNoOrder) {
        auto net = Network();
        net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
        net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
        EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
      }

      TEST_F(NetworkTest, GpuNetworkDenseFindOptimal) {
        auto net = Network();
        net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
        net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
        net.setOrder(true, "");
        EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
      }

      TEST_F(NetworkTest, GpuNetworkDenseOrderLine) {
        auto net = Network();
        net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "ORDER:(A,(B,C))", "TOUT: a,b;e"});
        net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
        EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
      }
      TEST_F(NetworkTest, GpuNetworkDenseSpecifiedOrder) {
        auto net = Network();
        net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
        net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
        net.setOrder(false, "(A,(B,C))");
        EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
      }
      TEST_F(NetworkTest, GpuNetworkDenseReuse) {
        auto net = Network();
        net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
        net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
        net.setOrder(false, "(A,(B,C))");
        EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
        net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnC, utdnB});
        EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
      }

      // The chain A(a,b,c) B(c,d) C(d,e) with entries given by closed-form expressions. All entries
      // are small multiples of 1/8, so every product and partial sum is exactly representable in
      // float as well as in double, and the expected values (plain loops over the same
      // expressions) do not depend on the contraction order.
      constexpr cytnx_uint64 kDimA = 3, kDimB = 4, kDimC = 5, kDimD = 6, kDimE = 2;

      bool IsComplexDtype(unsigned int dtype) {
        return dtype == Type.ComplexDouble || dtype == Type.ComplexFloat;
      }

      std::complex<double> ChainElemA(cytnx_uint64 a, cytnx_uint64 b, cytnx_uint64 c,
                                      bool is_complex) {
        const double re = (1.0 * a - 2.0 * b + 1.0 * c * c) / 8 - 0.375;
        const double im = (2.0 * a + 1.0 * b - 3.0 * c) / 8 + 0.125;
        return {re, is_complex ? im : 0.0};
      }

      std::complex<double> ChainElemB(cytnx_uint64 c, cytnx_uint64 d, bool is_complex) {
        const double re = (3.0 * c - 1.0 * d * d) / 8 + 0.25;
        const double im = (1.0 * c - 2.0 * d) / 8 - 0.5;
        return {re, is_complex ? im : 0.0};
      }

      std::complex<double> ChainElemC(cytnx_uint64 d, cytnx_uint64 e, bool is_complex) {
        const double re = (1.0 * d - 4.0 * e) / 8 + 0.125;
        const double im = (2.0 * e - 1.0 * d) / 8 + 0.375;
        return {re, is_complex ? im : 0.0};
      }

      // Result element (a,b,e) of the chain: sum over c and d of A(a,b,c) B(c,d) C(d,e), where
      // each tensor is complex or real on its own.
      std::complex<double> ChainExpected(cytnx_uint64 a, cytnx_uint64 b, cytnx_uint64 e,
                                         bool complex_a, bool complex_b, bool complex_c) {
        std::complex<double> sum = 0.0;
        for (cytnx_uint64 c = 0; c < kDimC; c++) {
          for (cytnx_uint64 d = 0; d < kDimD; d++) {
            sum += ChainElemA(a, b, c, complex_a) * ChainElemB(c, d, complex_b) *
                   ChainElemC(d, e, complex_c);
          }
        }
        return sum;
      }

      void SetElem(Tensor& tensor, const std::vector<cytnx_uint64>& locator,
                   std::complex<double> value) {
        if (tensor.dtype() == Type.ComplexDouble) {
          tensor.at<cytnx_complex128>(locator) = value;
        } else {
          tensor.at<cytnx_double>(locator) = value.real();
        }
      }

      // A CPU tensor of the given dtype and shape whose entries are elem(locator).
      Tensor MakeTensor(
        const std::vector<cytnx_uint64>& shape, unsigned int dtype,
        const std::function<std::complex<double>(const std::vector<cytnx_uint64>&, bool)>& elem) {
        const bool is_complex = IsComplexDtype(dtype);
        Tensor tensor = zeros(shape, is_complex ? Type.ComplexDouble : Type.Double);
        std::vector<cytnx_uint64> locator(shape.size(), 0);
        std::size_t total = 1;
        for (std::size_t extent : shape) total *= extent;
        for (std::size_t flat = 0; flat < total; flat++) {
          std::size_t rest = flat;
          for (std::size_t axis = shape.size(); axis-- > 0;) {
            locator[axis] = rest % shape[axis];
            rest /= shape[axis];
          }
          SetElem(tensor, locator, elem(locator, is_complex));
        }
        return tensor.astype(dtype);
      }

      // The three chain tensors on the GPU, labelled a,b,c / c,d / d,e.
      std::vector<UniTensor> MakeChain(unsigned int dtype_a, unsigned int dtype_b,
                                       unsigned int dtype_c) {
        Tensor A = MakeTensor({kDimA, kDimB, kDimC}, dtype_a,
                              [](const std::vector<cytnx_uint64>& loc, bool is_complex) {
                                return ChainElemA(loc[0], loc[1], loc[2], is_complex);
                              });
        Tensor B = MakeTensor({kDimC, kDimD}, dtype_b,
                              [](const std::vector<cytnx_uint64>& loc, bool is_complex) {
                                return ChainElemB(loc[0], loc[1], is_complex);
                              });
        Tensor C = MakeTensor({kDimD, kDimE}, dtype_c,
                              [](const std::vector<cytnx_uint64>& loc, bool is_complex) {
                                return ChainElemC(loc[0], loc[1], is_complex);
                              });
        return {UniTensor(A.to(Device.cuda), false, -1, {"a", "b", "c"}),
                UniTensor(B.to(Device.cuda), false, -1, {"c", "d"}),
                UniTensor(C.to(Device.cuda), false, -1, {"d", "e"})};
      }

      std::vector<UniTensor> MakeChain(unsigned int dtype) {
        return MakeChain(dtype, dtype, dtype);
      }

      using ExpectedFn =
        std::function<std::complex<double>(cytnx_uint64 a, cytnx_uint64 b, cytnx_uint64 e)>;

      ExpectedFn ChainExpectedFn(unsigned int dtype) {
        const bool is_complex = IsComplexDtype(dtype);
        return [is_complex](cytnx_uint64 a, cytnx_uint64 b, cytnx_uint64 e) {
          return ChainExpected(a, b, e, is_complex, is_complex, is_complex);
        };
      }

      // Checks dtype, device, labels, rowrank, shape and every element of a result with legs a, b
      // and e ordered as `labels`.
      void ExpectChainResult(const UniTensor& res, unsigned int dtype,
                             const std::vector<std::string>& labels, cytnx_int64 rowrank,
                             const ExpectedFn& expected_fn) {
        EXPECT_EQ(res.dtype(), dtype);
        EXPECT_EQ(res.device(), Device.cuda);
        EXPECT_EQ(res.labels(), labels);
        EXPECT_EQ(res.rowrank(), rowrank);

        const std::map<std::string, cytnx_uint64> dims = {{"a", kDimA}, {"b", kDimB}, {"e", kDimE}};
        std::vector<cytnx_uint64> shape;
        for (const std::string& label : labels) shape.push_back(dims.at(label));
        ASSERT_EQ(res.shape(), shape);

        const bool is_single = dtype == Type.ComplexFloat || dtype == Type.Float;
        const double tol = is_single ? 1e-5 : 1e-12;
        const Tensor block = res.to(Device.cpu).get_block().contiguous().astype(Type.ComplexDouble);
        const auto pos = [&labels](const std::string& label) {
          return std::find(labels.begin(), labels.end(), label) - labels.begin();
        };
        for (cytnx_uint64 a = 0; a < kDimA; a++) {
          for (cytnx_uint64 b = 0; b < kDimB; b++) {
            for (cytnx_uint64 e = 0; e < kDimE; e++) {
              std::vector<cytnx_uint64> locator(3);
              locator[pos("a")] = a;
              locator[pos("b")] = b;
              locator[pos("e")] = e;
              const std::complex<double> got = block.at<cytnx_complex128>(locator);
              const std::complex<double> expected = expected_fn(a, b, e);
              EXPECT_LE(std::abs(got - expected), tol)
                << "at (a,b,e)=(" << a << "," << b << "," << e << "): got " << got << ", expected "
                << expected;
            }
          }
        }
      }

      const std::vector<unsigned int> kFloatDtypes = {Type.ComplexDouble, Type.ComplexFloat,
                                                      Type.Double, Type.Float};

      // Double, Float and ComplexFloat used to be contracted as if they were ComplexDouble, and
      // the result had default labels and rowrank.
      TEST_F(NetworkTest, GpuNetworkDenseFloatDtypes) {
        for (unsigned int dtype : kFloatDtypes) {
          SCOPED_TRACE(Type.getname(dtype));
          auto net = Network();
          net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
          net.PutUniTensors({"A", "B", "C"}, MakeChain(dtype));
          ExpectChainResult(net.Launch(), dtype, {"a", "b", "e"}, 2, ChainExpectedFn(dtype));
        }
      }

      TEST_F(NetworkTest, GpuNetworkDenseFloatDtypesPermutedOutput) {
        for (unsigned int dtype : kFloatDtypes) {
          SCOPED_TRACE(Type.getname(dtype));
          auto net = Network();
          net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "ORDER:(A,(B,C))", "TOUT: e;b,a"});
          net.PutUniTensors({"A", "B", "C"}, MakeChain(dtype));
          ExpectChainResult(net.Launch(), dtype, {"e", "b", "a"}, 1, ChainExpectedFn(dtype));
        }
      }

      TEST_F(NetworkTest, GpuNetworkDenseFloatDtypesFindOptimal) {
        for (unsigned int dtype : kFloatDtypes) {
          SCOPED_TRACE(Type.getname(dtype));
          auto net = Network();
          net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
          net.PutUniTensors({"A", "B", "C"}, MakeChain(dtype));
          net.setOrder(true, "");
          ExpectChainResult(net.Launch(), dtype, {"a", "b", "e"}, 2, ChainExpectedFn(dtype));
        }
      }

      // cuTensorNet takes one dtype for the whole network. Mixed dtypes used to be read as the
      // dtype of the first tensor; they now take the generic path and promote like on the CPU.
      TEST_F(NetworkTest, GpuNetworkDenseMixedDtypes) {
        struct Case {
          unsigned int dtype_a, dtype_b, dtype_c, dtype_out;
        };
        const std::vector<Case> cases = {
          {Type.Double, Type.ComplexDouble, Type.Double, Type.ComplexDouble},
          {Type.ComplexDouble, Type.Double, Type.Double, Type.ComplexDouble},
          {Type.Double, Type.Float, Type.Double, Type.Double},
          {Type.Float, Type.ComplexFloat, Type.Float, Type.ComplexFloat},
        };
        for (const Case& c : cases) {
          SCOPED_TRACE(Type.getname(c.dtype_a) + " x " + Type.getname(c.dtype_b) + " x " +
                       Type.getname(c.dtype_c));
          auto net = Network();
          net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
          net.PutUniTensors({"A", "B", "C"}, MakeChain(c.dtype_a, c.dtype_b, c.dtype_c));
          const bool complex_a = IsComplexDtype(c.dtype_a);
          const bool complex_b = IsComplexDtype(c.dtype_b);
          const bool complex_c = IsComplexDtype(c.dtype_c);
          ExpectChainResult(net.Launch(), c.dtype_out, {"a", "b", "e"}, 2,
                            [=](cytnx_uint64 a, cytnx_uint64 b, cytnx_uint64 e) {
                              return ChainExpected(a, b, e, complex_a, complex_b, complex_c);
                            });
        }
      }

      // A diagonal UniTensor stores only its diagonal, which cuTensorNet used to read as if it
      // were the full matrix.
      TEST_F(NetworkTest, GpuNetworkDenseDiagonalInput) {
        for (unsigned int dtype : kFloatDtypes) {
          SCOPED_TRACE(Type.getname(dtype));
          const bool is_complex = IsComplexDtype(dtype);
          UniTensor A = MakeChain(dtype)[0];
          const auto diag_elem = [](cytnx_uint64 c, bool is_complex) {
            return ChainElemB(c, c + 1, is_complex);
          };
          Tensor diag =
            MakeTensor({kDimC}, dtype, [&](const std::vector<cytnx_uint64>& loc, bool is_complex) {
              return diag_elem(loc[0], is_complex);
            });
          UniTensor D(diag.to(Device.cuda), true, -1, {"c", "d"});
          Tensor c_block = MakeTensor({kDimC, kDimE}, dtype,
                                      [](const std::vector<cytnx_uint64>& loc, bool is_complex) {
                                        return ChainElemC(loc[0], loc[1], is_complex);
                                      });
          UniTensor C(c_block.to(Device.cuda), false, -1, {"d", "e"});

          auto net = Network();
          net.FromString({"A: a,b,c", "D: c,d", "C: d,e", "TOUT: a,b;e"});
          net.PutUniTensors({"A", "D", "C"}, {A, D, C});
          ExpectChainResult(net.Launch(), dtype, {"a", "b", "e"}, 2,
                            [&](cytnx_uint64 a, cytnx_uint64 b, cytnx_uint64 e) {
                              std::complex<double> sum = 0.0;
                              for (cytnx_uint64 c = 0; c < kDimC; c++) {
                                sum += ChainElemA(a, b, c, is_complex) * diag_elem(c, is_complex) *
                                       ChainElemC(c, e, is_complex);
                              }
                              return sum;
                            });
        }
      }

      // #958: the tensor-list form did not set up what the cuQuantum branch of Launch reads.
      TEST_F(NetworkTest, GpuNetworkStaticContract) {
        for (unsigned int dtype : kFloatDtypes) {
          SCOPED_TRACE(Type.getname(dtype));
          ExpectChainResult(Network::Contract(MakeChain(dtype), "a,b;e").Launch(), dtype,
                            {"a", "b", "e"}, 2, ChainExpectedFn(dtype));
        }
      }

      TEST_F(NetworkTest, GpuNetworkStaticContractSpecifiedOrder) {
        for (unsigned int dtype : kFloatDtypes) {
          SCOPED_TRACE(Type.getname(dtype));
          UniTensor res =
            Network::Contract(MakeChain(dtype), "e;b,a", {"A", "B", "C"}, "(A,(B,C))").Launch();
          ExpectChainResult(res, dtype, {"e", "b", "a"}, 1, ChainExpectedFn(dtype));
        }
      }

      // A closed network: every leg is contracted, so the result is a rank-0 UniTensor holding
      // sum over i and j of S1(i,j) S2(j,i).
      std::vector<UniTensor> MakeClosedPair(unsigned int dtype) {
        Tensor S1 = MakeTensor({kDimC, kDimD}, dtype,
                               [](const std::vector<cytnx_uint64>& loc, bool is_complex) {
                                 return ChainElemB(loc[0], loc[1], is_complex);
                               });
        Tensor S2 = MakeTensor({kDimD, kDimC}, dtype,
                               [](const std::vector<cytnx_uint64>& loc, bool is_complex) {
                                 return ChainElemA(loc[0], loc[1], 1, is_complex);
                               });
        return {UniTensor(S1.to(Device.cuda), false, -1, {"i", "j"}),
                UniTensor(S2.to(Device.cuda), false, -1, {"j", "i"})};
      }

      void ExpectClosedPairResult(const UniTensor& res, unsigned int dtype) {
        EXPECT_EQ(res.dtype(), dtype);
        EXPECT_EQ(res.device(), Device.cuda);
        EXPECT_TRUE(res.labels().empty());
        EXPECT_EQ(res.rowrank(), 0);
        ASSERT_TRUE(res.shape().empty());

        const bool is_complex = IsComplexDtype(dtype);
        std::complex<double> expected = 0.0;
        for (cytnx_uint64 i = 0; i < kDimC; i++) {
          for (cytnx_uint64 j = 0; j < kDimD; j++) {
            expected += ChainElemB(i, j, is_complex) * ChainElemA(j, i, 1, is_complex);
          }
        }
        const bool is_single = dtype == Type.ComplexFloat || dtype == Type.Float;
        const std::complex<double> got =
          res.to(Device.cpu).get_block().astype(Type.ComplexDouble).item<cytnx_complex128>();
        EXPECT_LE(std::abs(got - expected), is_single ? 1e-5 : 1e-12)
          << "got " << got << ", expected " << expected;
      }

      TEST_F(NetworkTest, GpuNetworkDenseScalarOutput) {
        for (unsigned int dtype : kFloatDtypes) {
          SCOPED_TRACE(Type.getname(dtype));
          auto net = Network();
          net.FromString({"S1: i,j", "S2: j,i", "TOUT:"});
          net.PutUniTensors({"S1", "S2"}, MakeClosedPair(dtype));
          ExpectClosedPairResult(net.Launch(), dtype);
        }
      }

      TEST_F(NetworkTest, GpuNetworkStaticContractScalarOutput) {
        for (unsigned int dtype : kFloatDtypes) {
          SCOPED_TRACE(Type.getname(dtype));
          ExpectClosedPairResult(Network::Contract(MakeClosedPair(dtype), "").Launch(), dtype);
        }
      }

      // Helper: Copy tensors to CPU. Contract them directly with Contract, and permute the open
      // legs into the requested TOUT order.
      static UniTensor BlockNetworkReferenceCPU(const UniTensor& A, const UniTensor& B,
                                                const UniTensor& C) {
        UniTensor a = A.to(Device.cpu).relabel({"a", "e"});
        UniTensor b = B.to(Device.cpu).relabel({"a", "c_", "d_", "h"});
        UniTensor c = C.to(Device.cpu).relabel({"e", "f_", "g_", "h"});
        UniTensor expected = Contract(Contract(b, c), a);
        expected.permute_({"c_", "d_", "f_", "g_"}, 2);
        expected.contiguous_();
        return expected;
      }

      // Block (symmetric) UniTensor network contraction on the GPU. Validate the GPU result against
      // a CPU contraction of the same tensors.
      TEST_F(NetworkTest, GpuNetworkBlockNoOrder) {
        random::uniform_(bkut1, -1., 1., 1);
        random::uniform_(bkut2, -1., 1., 2);
        random::uniform_(bkut3, -1., 1., 3);

        auto net = Network();
        net.FromString({"A: a,e", "B: a,c_,d_,h", "C: e,f_,g_,h", "TOUT: c_,d_;f_,g_"});
        net.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3});
        UniTensor res = net.Launch();
        EXPECT_EQ(res.uten_type(), UTenType.Block);
        EXPECT_EQ(res.device(), Device.cuda);

        UniTensor res_cpu = res.to(Device.cpu);
        res_cpu.contiguous_();
        EXPECT_TRUE(
          AreNearlyEqUniTensor(res_cpu, BlockNetworkReferenceCPU(bkut1, bkut2, bkut3), 1e-8));
      }

      TEST_F(NetworkTest, GpuNetworkBlockSpecifiedOrder) {
        random::uniform_(bkut1, -1., 1., 1);
        random::uniform_(bkut2, -1., 1., 2);
        random::uniform_(bkut3, -1., 1., 3);

        auto net = Network();
        net.FromString(
          {"A: a,e", "B: a,c_,d_,h", "C: e,f_,g_,h", "ORDER:(A,(B,C))", "TOUT: c_,d_;f_,g_"});
        net.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3});
        UniTensor res = net.Launch();

        UniTensor res_cpu = res.to(Device.cpu);
        res_cpu.contiguous_();
        EXPECT_TRUE(
          AreNearlyEqUniTensor(res_cpu, BlockNetworkReferenceCPU(bkut1, bkut2, bkut3), 1e-8));
      }

      // setOrder(true, "") on a non-dense network
      TEST_F(NetworkTest, GpuNetworkBlockFindOptimal) {
        random::uniform_(bkut1, -1., 1., 1);
        random::uniform_(bkut2, -1., 1., 2);
        random::uniform_(bkut3, -1., 1., 3);

        auto net = Network();
        net.FromString({"A: a,e", "B: a,c_,d_,h", "C: e,f_,g_,h", "TOUT: c_,d_;f_,g_"});
        net.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3});
        net.setOrder(true, "");
        UniTensor res = net.Launch();

        UniTensor res_cpu = res.to(Device.cpu);
        res_cpu.contiguous_();
        EXPECT_TRUE(
          AreNearlyEqUniTensor(res_cpu, BlockNetworkReferenceCPU(bkut1, bkut2, bkut3), 1e-8));
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
