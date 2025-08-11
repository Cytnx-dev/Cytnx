#ifndef GPU_TEST_TOOLS_H_
#define GPU_TEST_TOOLS_H_

#include <string>
#include <vector>
#include <random>

#include "gtest/gtest.h"

#include "Device.hpp"
#include "Tensor.hpp"
#include "Type.hpp"
#include "UniTensor.hpp"

// This file contains some function we may usually use in the unit test
// such as the data initialization and comparison.
namespace cytnx {
  namespace TestTools {

    // test error message trace
    class TestFailMsg {
     private:
      std::string test_case;
      std::vector<std::string> fail_msgs;

     public:
      // TestFailMsg();
      void Init(const std::string& _test_case) {
        test_case = _test_case;
        fail_msgs.clear();
      }
      void AppendMsg(const std::string& fail_msg, const std::string& func_name, const int line);
      void AppendMsg(const std::string& fail_msg) { fail_msgs.push_back(fail_msg); }
      bool is_empty() { return !(test_case.empty() && fail_msgs.empty()); };
      std::string TraceFailMsgs();
    };

    static std::vector<unsigned int> dtype_list = {
      // Type.Void,
      Type.ComplexDouble, Type.ComplexFloat, Type.Double, Type.Float,  Type.Int64, Type.Uint64,
      Type.Int32,         Type.Uint32,       Type.Int16,  Type.Uint16, Type.Bool};

    static std::vector<int> device_list = {
      Device.cpu,
      // Device.cuda,  //currently cuda version still not implement
    };

    // Tensor tools

    // given the tensor T with shape and dtype has been initialzed, set its data as random uniform.
    void InitTensorUniform(Tensor& T, unsigned int rand_seed = 0);
    void InitTensorUniform(std::vector<Tensor>& T, unsigned int rand_seed = 0);

    bool AreNearlyEqTensor(const Tensor& T1, const Tensor& T2, const cytnx_double tol = 0);
    bool AreEqTensor(const Tensor& T1, const Tensor& T2);

    bool AreNearlyEqUniTensor(const UniTensor& Ut1, const UniTensor& Ut2,
                              const cytnx_double tol = 0);
    bool AreEqUniTensor(const UniTensor& Ut1, const UniTensor& Ut2);

    bool AreElemSame(const Tensor& T1, const std::vector<cytnx_uint64>& idices1, const Tensor& T2,
                     const std::vector<cytnx_uint64>& idices2);

    void InitUniTensorUniform(UniTensor& UT, unsigned int rand_seed = 0);

    /**
     * Helper function for the EXPECT_NUMBER_EQ assertion.
     * This function compares floating-point numbers, considering their ULP (Units in Last Place)
     * distance. If the distance between two floating-point numbers is less than 4 ULPs, they are
     * considered equal.
     */
    template <typename T1, typename T2>
    ::testing::AssertionResult AreEqNumber(const char* expr1, const char* expr2, T1 value1,
                                           T2 value2) {
      static_assert(std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>);
      // Prevent unintentional comparison between float and double.
      static_assert(std::is_floating_point_v<T1> && std::is_floating_point_v<T2>
                      ? std::is_same_v<T1, T2>
                      : true);
      if constexpr (std::is_integral_v<T1> && std::is_integral_v<T2>) {
        return ::testing::internal::EqHelper::Compare(expr1, expr2, value1, value2);
      }
      if constexpr (std::is_floating_point_v<T1>) {
        return ::testing::internal::CmpHelperFloatingPointEQ<T1>(expr1, expr2, value1, value2);
      }
      // std::is_floating_point_v<T2>
      return ::testing::internal::CmpHelperFloatingPointEQ<T2>(expr1, expr2, value1, value2);
    }

    /**
     * Complex floating-point numbers are compared based on the ULP distances of their real and
     * imaginary parts, rather than using the Euclidean distance between the two complex numbers.
     */
    template <typename T>
    ::testing::AssertionResult AreEqNumber(const char* expr1, const char* expr2,
                                           const std::complex<T>& value1,
                                           const std::complex<T>& value2) {
      ::testing::internal::FloatingPoint<T> float1_real(value1.real()), float1_imag(value1.imag()),
        float2_real(value2.real()), float2_imag(value2.imag());
      if (float1_real.AlmostEquals(float2_real) && float1_imag.AlmostEquals(float2_imag)) {
        return ::testing::AssertionSuccess();
      }

      std::stringstream value1_stream;
      value1_stream.precision(std::numeric_limits<T>::digits10 + 2);
      value1_stream << value1.real() << "+" << value1.imag() << "j";

      std::stringstream value2_stream;
      value2_stream.precision(std::numeric_limits<T>::digits10 + 2);
      value2_stream << value2.real() << "+" << value2.imag() << "j";
      return ::testing::internal::EqFailure(expr1, expr2, value1_stream.str(), value2_stream.str(),
                                            false);
    }

    std::vector<std::vector<cytnx_uint64>> GenerateTestShapes(
      cytnx_uint64 dim, cytnx_uint64 min_size = 1, cytnx_uint64 max_size = 1024,
      cytnx_uint64 num_shapes = 10, cytnx_bool include_edge_case = true, cytnx_uint32 seed = 0);

  }  // namespace TestTools
}  // namespace cytnx

/**
 * This assertion handles all supported data types and is useful in type-parameterized tests.
 * The key difference between EXPECT_NUMBER_EQ and gtest's EXPECT_EQ is that EXPECT_NUMBER_EQ
 * compares floating-point numbers based on their ULP (Units in Last Place) distance.
 */
#define EXPECT_NUMBER_EQ(value1, value2) \
  EXPECT_PRED_FORMAT2(cytnx::TestTools::AreEqNumber, value1, value2)

#endif  // GPU_TEST_TOOLS_H_
