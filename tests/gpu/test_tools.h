#ifndef _H_test_tools
#define _H_test_tools

#include "cytnx.hpp"
#include <gtest/gtest.h>

// this file contains some function we may usually use in the unit test
//   such as the data initialization and comparison.

using namespace cytnx;

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

  bool AreNearlyEqUniTensor(const UniTensor& Ut1, const UniTensor& Ut2, const cytnx_double tol = 0);
  bool AreEqUniTensor(const UniTensor& Ut1, const UniTensor& Ut2);

  bool AreElemSame(const Tensor& T1, const std::vector<cytnx_uint64>& idices1, const Tensor& T2,
                   const std::vector<cytnx_uint64>& idices2);

  void InitUniTensorUniform(UniTensor& UT, unsigned int rand_seed = 0);
}  // namespace TestTools

#endif
