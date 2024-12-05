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

  template <typename, typename>
  struct TypeCombinationsImpl;

  template <typename FirstType, typename... SecondTypes>
  struct TypeCombinationsImpl<FirstType, std::tuple<SecondTypes...>> {
    using type = std::tuple<std::pair<FirstType, SecondTypes>...>;
  };

  template <typename... FirstTypes, typename... SecondTypes>
  struct TypeCombinationsImpl<std::tuple<FirstTypes...>, std::tuple<SecondTypes...>> {
    using type = decltype(std::tuple_cat(
      std::declval<
        typename TypeCombinationsImpl<FirstTypes, std::tuple<SecondTypes...>>::type>()...));
  };

  /**
   * @brief Generate pairs of all combinations of the types in the given tuples.
   *
   * @tparam FirstTuple a tuple composed of the types which will be the first type of the pair
   * @tparam SecondTuple a tuple composed of the types which will be the second type of the pair
   *
   *
   * @code
   * ```cpp
   *
   * static_assert(
   *     std::is_same_v<
   *         TypeCombinations<std::tuple<int, std::string>,
   *                          std::tuple<int, double, std::string>>,
   *         std::tuple<std::pair<int, int>, std::pair<int, double>,
   *                    std::pair<int, std::string>, std::pair<std::string, int>,
   *                    std::pair<std::string, double>,
   *                    std::pair<std::string, std::string>>>);
   *
   * ```
   * @endcode
   */
  template <typename FirstTuple, typename SecondTuple>
  using TypeCombinations = typename TypeCombinationsImpl<FirstTuple, SecondTuple>::type;

  template <typename... TypesInTuple>
  constexpr auto TupleToTestTypesHelper(std::tuple<TypesInTuple...>)
    -> testing::Types<TypesInTuple...>;

  /**
   * @brief Generate pairs of all combinations of the types in the given tuples.
   *
   * @tparam FirstTuple a tuple composed of the types which will be the first type of the pair
   * @tparam SecondTuple a tuple composed of the types which will be the second type of the pair
   *
   *
   * @code
   * ```cpp
   *
   * static_assert(
   *     std::is_same_v<
   *         TypeCombinations<std::tuple<int, std::string>,
   *                          std::tuple<int, double, std::string>>,
   *         testing::Types<std::pair<int, int>,
   *                        std::pair<int, double>,
   *                        std::p<air<int, std::string>,
   *                        std::pair<std::string, int>,
   *                        std::pair<std::string, double>,
   *                        std::pair<std::string, std::string>>>);
   *
   * ```
   * @endcode
   */
  template <typename FirstTuple, typename SecondTuple>
  using TestTypeCombinations =
    decltype(TupleToTestTypesHelper(std::declval<TypeCombinations<FirstTuple, SecondTuple>>()));

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
