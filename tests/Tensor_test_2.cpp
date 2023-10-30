#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "cytnx.hpp"

using namespace cytnx;
using namespace testing;

typedef std::vector<std::complex<double>> ValuesList;
typedef Accessor ac;

namespace MyTensorTest {

  static std::vector<unsigned int> dtype_list = {
    // Type.Void,
    Type.ComplexDouble, Type.ComplexFloat, Type.Double, Type.Float,  Type.Int64, Type.Uint64,
    Type.Int32,         Type.Uint32,       Type.Int16,  Type.Uint16, Type.Bool};

  static std::vector<int> device_list = {
    Device.cpu,
    // Device.cuda,  //need to be test on cuda support environment
  };

  static std::map<unsigned int, std::string> dtype_str_map = {
    {Type.Void, "Void"},
    {Type.ComplexDouble, "Complex Double (Complex Float64)"},
    {Type.ComplexFloat, "Complex Float (Complex Float32)"},
    {Type.Double, "Double (Float64)"},
    {Type.Float, "Float (Float32)"},
    {Type.Int64, "Int64"},
    {Type.Uint64, "Uint64"},
    {Type.Int32, "Int32"},
    {Type.Uint32, "Uint32"},
    {Type.Int16, "Int16"},
    {Type.Uint16, "Uint16"},
    {Type.Bool, "Bool"}};

  static std::map<int, std::string> device_str_map = {{Device.cpu, "cytnx device: CPU"},
                                                      {Device.cuda, "cytnx device: CUDA"}};

  // declaration of methods
  void CheckTensStruct(const Tensor& T, unsigned int expect_type, int expect_device,
                       std::vector<unsigned int> expect_shape);
  void tens3test(const Tensor& T, ValuesList values_array);
  void tens2test(const Tensor& T, ValuesList values_array);

  // check tensor struct
  void CheckTensStruct(const Tensor& T, unsigned int expect_type, int expect_device,
                       std::vector<unsigned int> expect_shape) {
    EXPECT_EQ(T.dtype(), expect_type);
    EXPECT_EQ(T.device(), expect_device);
    // check shape
    auto tens_shape = T.shape();  // std::vector
    size_t shape_size = tens_shape.size();
    EXPECT_EQ(shape_size, expect_shape.size());
    for (size_t i = 0; i < shape_size; ++i) {
      EXPECT_EQ(tens_shape[i], expect_shape[i]);
    }
  }

  void tens3test(const Tensor& T, ValuesList values_array) {
    auto tens_shape = T.shape();  // std::vector
    ASSERT_EQ(tens_shape.size(), 3);
    unsigned int i_num = tens_shape[0], j_num = tens_shape[1], k_num = tens_shape[2], index = 0;
    for (auto i = 0u; i < i_num; ++i) {
      for (auto j = 0u; j < j_num; ++j) {
        for (auto k = 0u; k < k_num; ++k) {
          switch (T.dtype()) {
            case Type.Void:
              break;
            case Type.ComplexDouble: {
              auto tens_val = T.at<std::complex<double>>(i, j, k);
              EXPECT_EQ(tens_val, std::complex<double>(values_array[index]));
              break;
            }
            case Type.ComplexFloat: {
              auto tens_val = T.at<std::complex<float>>(i, j, k);
              EXPECT_EQ(tens_val, std::complex<float>(values_array[index]));
              break;
            }
            case Type.Double: {
              auto tens_val = T.at<double>(i, j, k);
              EXPECT_EQ(tens_val, double(values_array[index].real()));
              break;
            }
            case Type.Float: {
              auto tens_val = T.at<float>(i, j, k);
              EXPECT_EQ(tens_val, float(values_array[index].real()));
              break;
            }
            case Type.Int64: {
              auto tens_val = T.at<int64_t>(i, j, k);
              EXPECT_EQ(tens_val, int64_t(values_array[index].real()));
              break;
            }
            case Type.Uint64: {
              auto tens_val = T.at<uint64_t>(i, j, k);
              EXPECT_EQ(tens_val, uint64_t(values_array[index].real()));
              break;
            }
            case Type.Int32: {
              auto tens_val = T.at<int32_t>(i, j, k);
              EXPECT_EQ(tens_val, int32_t(values_array[index].real()));
              break;
            }
            case Type.Uint32: {
              auto tens_val = T.at<uint32_t>(i, j, k);
              EXPECT_EQ(tens_val, uint32_t(values_array[index].real()));
              break;
            }
            case Type.Int16: {
              auto tens_val = T.at<int16_t>(i, j, k);
              EXPECT_EQ(tens_val, int16_t(values_array[index].real()));
              break;
            }
            case Type.Uint16: {
              auto tens_val = T.at<uint16_t>(i, j, k);
              EXPECT_EQ(tens_val, uint16_t(values_array[index].real()));
              break;
            }
            case Type.Bool: {
              auto tens_val = T.at<bool>(i, j, k);
              EXPECT_EQ(tens_val, bool(values_array[index].real()));
              break;
            }
            default:
              FAIL();  // never
          }
          index++;
        }
      }
    }
  }

  void tens2test(const Tensor& T, ValuesList values_array) {
    auto tens_shape = T.shape();  // std::vector
    ASSERT_EQ(tens_shape.size(), 2);
    unsigned int i_num = tens_shape[0], j_num = tens_shape[1], index = 0;
    for (auto i = 0u; i < i_num; ++i) {
      for (auto j = 0u; j < j_num; ++j) {
        switch (T.dtype()) {
          case Type.Void:
            break;
          case Type.ComplexDouble: {
            auto tens_val = T.at<std::complex<double>>(i, j);
            EXPECT_EQ(tens_val, std::complex<double>(values_array[index]));
            break;
          }
          case Type.ComplexFloat: {
            auto tens_val = T.at<std::complex<float>>(i, j);
            EXPECT_EQ(tens_val, std::complex<float>(values_array[index]));
            break;
          }
          case Type.Double: {
            auto tens_val = T.at<double>(i, j);
            EXPECT_EQ(tens_val, double(values_array[index].real()));
            break;
          }
          case Type.Float: {
            auto tens_val = T.at<float>(i, j);
            EXPECT_EQ(tens_val, float(values_array[index].real()));
            break;
          }
          case Type.Int64: {
            auto tens_val = T.at<int64_t>(i, j);
            EXPECT_EQ(tens_val, int64_t(values_array[index].real()));
            break;
          }
          case Type.Uint64: {
            auto tens_val = T.at<uint64_t>(i, j);
            EXPECT_EQ(tens_val, uint64_t(values_array[index].real()));
            break;
          }
          case Type.Int32: {
            auto tens_val = T.at<int32_t>(i, j);
            EXPECT_EQ(tens_val, int32_t(values_array[index].real()));
            break;
          }
          case Type.Uint32: {
            auto tens_val = T.at<uint32_t>(i, j);
            EXPECT_EQ(tens_val, uint32_t(values_array[index].real()));
            break;
          }
          case Type.Int16: {
            auto tens_val = T.at<int16_t>(i, j);
            EXPECT_EQ(tens_val, int16_t(values_array[index].real()));
            break;
          }
          case Type.Uint16: {
            auto tens_val = T.at<uint16_t>(i, j);
            EXPECT_EQ(tens_val, uint16_t(values_array[index].real()));
            break;
          }
          case Type.Bool: {
            auto tens_val = T.at<bool>(i, j);
            EXPECT_EQ(tens_val, bool(values_array[index].real()));
            break;
          }
          default:
            FAIL();  // never
        }
        index++;
      }
    }
  }

  TEST(Tensor, dtype) {
    // dtype
    for (auto assign_dtype : dtype_list) {
      Tensor T1({2, 2, 2}, assign_dtype);
      EXPECT_EQ(T1.dtype(), assign_dtype);
    }

    // dtype_str
    for (auto assign_dtype : dtype_list) {
      Tensor T2({2, 2, 2}, assign_dtype);
      std::string dtype_str = dtype_str_map[assign_dtype];
      EXPECT_EQ(T2.dtype_str(), dtype_str);
    }
  }

  TEST(Tensor, device) {
    // dtype
    for (auto assign_device : device_list) {
      Tensor T1({2, 2, 2}, Type.Double, assign_device);
      EXPECT_EQ(T1.device(), assign_device);
    }

    // dtype_str
    for (auto assign_device : device_list) {
      Tensor T2({2, 2, 2}, Type.Double, assign_device);
      std::string device_str = device_str_map[assign_device];
      EXPECT_EQ(T2.device_str(), device_str);
    }
  }

  TEST(Tensor, constructor) {
    // default
    Tensor T1 = Tensor();
    CheckTensStruct(T1, Type.Void, Device.cpu, {});

    // shape assigned
    Tensor T2({2, 2, 2});
    CheckTensStruct(T2, Type.Double, Device.cpu, {2, 2, 2});
    ValuesList V2{0, 0, 0, 0, 0, 0, 0, 0};
    tens3test(T2, V2);

    // device assigned
    for (auto& assign_device : device_list) {
      for (auto& assign_dtype : dtype_list) {
        Tensor T3({2, 2, 2}, assign_dtype, assign_device);
        CheckTensStruct(T3, assign_dtype, assign_device, {2, 2, 2});
        ValuesList V1{0, 0, 0, 0, 0, 0, 0, 0};
        tens3test(T3, V1);
      }
    }
  }

  TEST(Tensor, init) {
    for (auto& assign_device : device_list) {
      for (auto& assign_dtype : dtype_list) {
        Tensor T1;
        T1.Init({2, 2, 2}, assign_dtype, assign_device);
        CheckTensStruct(T1, assign_dtype, assign_device, {2, 2, 2});
        ValuesList V1{0, 0, 0, 0, 0, 0, 0, 0};
        tens3test(T1, V1);
      }
    }
  }

  // real
  TEST(Tensor, real) {
    Tensor T1 = Tensor({2, 2}, Type.ComplexDouble);
    T1(0, 0) = std::complex<double>(9, 1);
    T1 = T1.real();
    double value = T1(0, 0).item<double>();  // 9
    EXPECT_EQ(double(9), value);
  }

  // imag
  TEST(Tensor, imag) {
    Tensor T1 = Tensor({2, 2}, Type.ComplexDouble);
    T1(0, 0) = std::complex<double>(9, 1);
    T1 = T1.imag();
    double value = T1(0, 0).item<double>();  // 1
    EXPECT_EQ(double(1), value);
  }

  // fill
  TEST(Tensor, fill) {
    Tensor T1 = Tensor({2, 2});
    T1.fill(9);
    ValuesList V1{9, 9, 9, 9};
    tens2test(T1, V1);
  }

  // equiv
  TEST(Tensor, equiv) {
    // equal shape
    Tensor A1 = Tensor({2, 2});
    Tensor B1 = Tensor({2, 2});
    bool equal = A1.equiv(B1);
    EXPECT_TRUE(equal);

    // inequal shape
    Tensor A2 = Tensor({2, 2});
    Tensor B2 = Tensor({3, 3});
    equal = A2.equiv(B2);
    EXPECT_FALSE(equal);
  }

  TEST(Tensor, zero) {
    // only shape is assigned
    Tensor T1 = zeros({2, 2, 2});
    CheckTensStruct(T1, Type.Double, Device.cpu, {2, 2, 2});
    ValuesList V1{0, 0, 0, 0, 0, 0, 0, 0};
    tens3test(T1, V1);

    // dtype device assigned
    for (auto type : dtype_list) {
      for (auto device : device_list) {
        Tensor T2 = zeros({2, 2, 2}, type, device);
        CheckTensStruct(T2, type, device, {2, 2, 2});
        ValuesList V2{0, 0, 0, 0, 0, 0, 0, 0};
        tens3test(T2, V2);
      }
    }
  }

  // one declaration
  TEST(Tensor, one) {
    // only shape is assigned
    Tensor T1 = ones({1, 2, 3});
    CheckTensStruct(T1, Type.Double, Device.cpu, {1, 2, 3});
    ValuesList V1{1, 1, 1, 1, 1, 1};
    tens3test(T1, V1);

    // dtype, device assigned
    for (auto type : dtype_list) {
      for (auto device : device_list) {
        Tensor T2 = ones({1, 2, 3}, type, device);
        CheckTensStruct(T2, type, device, {1, 2, 3});
        ValuesList V2{1, 1, 1, 1, 1, 1};
        tens3test(T2, V2);
      }
    }
  }

  TEST(Tensor, arange) {
    // arrange with only number of elements
    Tensor T1 = arange(6).reshape({1, 2, 3});
    CheckTensStruct(T1, Type.Double, Device.cpu, {1, 2, 3});
    ValuesList V1{0, 1, 2, 3, 4, 5};
    tens3test(T1, V1);

    // arrange with number of elements and start value
    Tensor T2 = arange(2, 8).reshape({1, 2, 3});  // from 2 to 7
    CheckTensStruct(T2, Type.Double, Device.cpu, {1, 2, 3});
    ValuesList V2{2, 3, 4, 5, 6, 7};
    tens3test(T2, V2);

    // arange with number of element, start values, and step specification
    Tensor T3 = arange(0, 12, 2).reshape({1, 2, 3});  // 0 2 4 6 8 10
    CheckTensStruct(T3, Type.Double, Device.cpu, {1, 2, 3});
    ValuesList V3{0, 2, 4, 6, 8, 10};
    tens3test(T3, V3);
  }

  TEST(Tensor, storage) {
    // from storage reference
    auto S1 = Storage(6);
    auto T1 = Tensor::from_storage(S1).reshape({1, 2, 3});
    CheckTensStruct(T1, Type.Double, Device.cpu, {1, 2, 3});
    ValuesList V1(6, 0);
    tens3test(T1, V1);

    // from storage with clone storage
    auto S2 = Storage(6);
    auto T2 = Tensor::from_storage(S2.clone()).reshape({1, 2, 3});  // different memory address
    CheckTensStruct(T2, Type.Double, Device.cpu, {1, 2, 3});
    ValuesList V2(6, 0);
    tens3test(T2, V2);

    // access element with ()
    Tensor A3 = arange(24).reshape(2, 3, 4);
    auto B3 = A3(0, ":", "1:4:2");
    CheckTensStruct(B3, Type.Double, Device.cpu, {3, 2});
  }

  TEST(Tensor, item) {
    for (auto type : dtype_list) {
      Tensor T = ones({2, 3}, type);
      auto shape = T.shape();
      for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
          auto element = T(i, j);

          // check item with different type
          switch (type) {
            case Type.Void:
              break;
            case Type.ComplexDouble: {
              std::complex<double> value = element.item<std::complex<double>>();
              EXPECT_EQ(value, std::complex<double>(1));
              break;
            }
            case Type.ComplexFloat: {
              std::complex<float> value = element.item<std::complex<float>>();
              EXPECT_EQ(value, std::complex<float>(1));
              break;
            }
            case Type.Double: {
              double value = element.item<double>();
              EXPECT_EQ(value, double(1));
              break;
            }
            case Type.Float: {
              float value = element.item<float>();
              EXPECT_EQ(value, float(1));
              break;
            }
            case Type.Int64: {
              int64_t value = element.item<int64_t>();
              EXPECT_EQ(value, int64_t(1));
              break;
            }
            case Type.Uint64: {
              uint64_t value = element.item<uint64_t>();
              EXPECT_EQ(value, uint64_t(1));
              break;
            }
            case Type.Int32: {
              int32_t value = element.item<int32_t>();
              EXPECT_EQ(value, int32_t(1));
              break;
            }
            case Type.Uint32: {
              uint32_t value = element.item<uint32_t>();
              EXPECT_EQ(value, uint32_t(1));
              break;
            }
            case Type.Int16: {
              int16_t value = element.item<int16_t>();
              EXPECT_EQ(value, int16_t(1));
              break;
            }
            case Type.Uint16: {
              uint16_t value = element.item<uint16_t>();
              EXPECT_EQ(value, uint16_t(1));
              break;
            }
            case Type.Bool: {
              bool value = element.item<bool>();
              EXPECT_EQ(value, bool(1));
              break;
            }
            default:
              FAIL();  // never
          }
        }
      }
    }
  }

  TEST(Tensor, set) {
    // set element with slice
    auto A1 = arange(9).reshape(3, 3);
    // [0, 1, 2], [3, 4, 5], [6, 7, 8]
    auto B1 = zeros({3, 2});
    // [0 0], [0 0], [0 0]
    A1(":", ":2") = B1;  // [0, 0, 3], [0, 0, 6], [0, 0, 9]
    ValuesList V1{0, 0, 2, 0, 0, 5, 0, 0, 8};
    tens2test(A1, V1);
    CheckTensStruct(A1, Type.Double, Device.cpu, {3, 3});
  }

  TEST(Tensor, accessor) {
    // accessor_get_mlevel
    Tensor A1 = arange(24).reshape(2, 3, 4);
    auto B1 = A1[{ac(0), ac::all(), ac::range(1, 4, 2)}];  // [1,3],[5,7],[9,11]
    CheckTensStruct(B1, Type.Double, Device.cpu, {3, 2});
    ValuesList V1{1, 3, 5, 7, 9, 11};
    tens2test(B1, V1);

    // accessor_get_llevel
    Tensor A2 = arange(24).reshape(2, 3, 4);
    auto B2 = A2.get({ac(0), ac::all(), ac::range(1, 4, 2)});  // [1,3],[5,7],[9,11]
    CheckTensStruct(B2, Type.Double, Device.cpu, {3, 2});
    ValuesList V2{1, 3, 5, 7, 9, 11};
    tens2test(B2, V2);

    // accessor_set_mlevel
    auto A3 = arange(9).reshape(3, 3);
    // [1, 2, 3], [4, 5, 6], [7, 8, 9]
    auto B3 = zeros({2});
    // [0 0], [0 0], [0 0]
    A3[{ac(0), ac::range(0, 2)}] = B3;  // [0, 0, 3]
    ValuesList V3{0, 0, 2, 3, 4, 5, 6, 7, 8};
    tens2test(A3, V3);
    CheckTensStruct(A3, Type.Double, Device.cpu, {3, 3});

    // accessor_set_llevel
    auto A4 = arange(9).reshape(3, 3);
    // [0, 1, 2], [3, 4, 5], [6, 7, 8]
    auto B4 = zeros({2});
    // [0 0], [0 0], [0 0]
    A4.set({ac(0), ac::range(0, 2)}, B4);  // [0, 0, 2]
    ValuesList V4{0, 0, 2, 3, 4, 5, 6, 7, 8};
    tens2test(A4, V4);
    CheckTensStruct(A4, Type.Double, Device.cpu, {3, 3});
  }

  // type_conversion
  TEST(Tensor, type_conversion) {
    Tensor A1({1, 2, 2});
    // type conversion
    for (auto type : dtype_list) {
      Tensor B1 = A1.astype(type);
      CheckTensStruct(B1, type, Device.cpu, {1, 2, 2});
      ValuesList V1{0, 0, 0, 0, 0, 0};
      tens3test(B1, V1);
    }
  }

  // device conversion
  TEST(Tensor, device_conversion) {
    Tensor T1({2, 2, 2});
    for (auto device : device_list) {
      Tensor T_ndevice = T1.to(device);
      CheckTensStruct(T_ndevice, Type.Double, device, {2, 2, 2});
      ValuesList V1{0, 0, 0, 0, 0, 0, 0, 0};
      tens3test(T_ndevice, V1);
    }
  }

  // permute
  TEST(Tensor, permute) {
    Tensor A1 = arange(12).reshape({2, 2, 3});
    // [[0,1,2], [3,4,5]],[[6,7,8], [9,10,11]] | shape = [2,2,3]
    Tensor B1 = A1.permute({1, 2, 0});
    // [[0,6],[1,7],[2,8]], [[3,9], [4,10], [5,11]] | shape = [2,3,2]
    ValuesList V1{0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11};
    CheckTensStruct(B1, A1.dtype(), A1.device(), {2, 3, 2});
    tens3test(B1, V1);
  }

  // permute_
  TEST(Tensor, permute_) {
    Tensor A1 = arange(12).reshape({2, 2, 3}).permute_(1, 2, 0);
    // [[0,6],[1,7],[2,8]], [[3,9], [4,10], [5,11]] | shape = [2,3,2]
    ValuesList V1{0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11};
    CheckTensStruct(A1, Type.Double, Device.cpu, {2, 3, 2});
    tens3test(A1, V1);
    EXPECT_FALSE(A1.is_contiguous());
    A1.contiguous_();
    EXPECT_TRUE(A1.is_contiguous());
  }

  // reshape
  TEST(Tensor, reshape_test) {
    Tensor A1 = arange(6).reshape({2, 3});
    CheckTensStruct(A1, Type.Double, Device.cpu, {2, 3});
    ValuesList V1{0, 1, 2, 3, 4, 5};
    tens2test(A1, V1);
  }

  TEST(Tensor, plus_constant) {
    // plus constant
    Tensor A1 = ones({1, 2}, Type.Int32);
    Tensor B1 = A1 + 4;
    ValuesList V1_1{5, 5, 5, 5};
    CheckTensStruct(B1, Type.Int32, Device.cpu, {1, 2});
    tens2test(B1, V1_1);

    Tensor C1 = A1 + std::complex<double>(0, 7);
    ValuesList V1_2{std::complex<double>(1, 7), std::complex<double>(1, 7),
                    std::complex<double>(1, 7), std::complex<double>(1, 7)};
    CheckTensStruct(C1, Type.ComplexDouble, Device.cpu, {1, 2});
    tens2test(C1, V1_2);

    Tensor D1 = A1 + double(2.2);
    ValuesList V1_3{3.2, 3.2, 3.2, 3.2};
    CheckTensStruct(D1, Type.Double, Device.cpu, {1, 2});
    tens2test(D1, V1_3);
  }

  TEST(Tensor, plus_tensor) {
    // plus tensor
    auto A2 = ones({1, 2});
    auto B2 = ones({1, 2});
    auto C2 = A2 + B2;
    ValuesList V2{2, 2};
    CheckTensStruct(C2, Type.Double, Device.cpu, {1, 2});
    tens2test(C2, V2);
  }

  TEST(Tensor, inline_plus) {
    // +=
    auto A3 = ones({2, 2});
    auto B3 = ones({2, 2});
    A3 += B3;
    ValuesList V3{2, 2, 2, 2};
    CheckTensStruct(A3, Type.Double, Device.cpu, {2, 2});
    tens2test(A3, V3);
  }

  TEST(Tensor, add) {
    // add
    auto A4 = ones({1, 2});
    auto B4 = ones({1, 2});
    auto C4 = A4.Add(B4);
    ValuesList V4{2, 2};
    CheckTensStruct(C4, Type.Double, Device.cpu, {1, 2});
    tens2test(C4, V4);

    // add_
    auto A5 = ones({1, 2});
    auto B5 = ones({1, 2});
    A5.Add_(B5);
    ValuesList V5{2, 2};
    CheckTensStruct(A5, Type.Double, Device.cpu, {1, 2});
    tens2test(A5, V5);
  }

  TEST(Tensor, product) {
    // *
    auto A1 = arange(12).reshape(3, 4);
    auto B1 = ones({3, 4}) * 4;
    auto C1 = A1 * B1;
    ValuesList V1{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44};
    CheckTensStruct(C1, Type.Double, Device.cpu, {3, 4});
    tens2test(C1, V1);

    // *=
    auto A2 = arange(12).reshape(3, 4);
    auto B2 = ones({3, 4}) * 4;
    A2 *= B2;
    ValuesList V2{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44};
    CheckTensStruct(A2, Type.Double, Device.cpu, {3, 4});
    tens2test(A2, V2);

    // MUL
    auto A3 = arange(12).reshape(3, 4);
    auto B3 = ones({3, 4}) * 4;
    auto C3 = A3.Mul(B1);
    ValuesList V3{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44};
    CheckTensStruct(C3, Type.Double, Device.cpu, {3, 4});
    tens2test(C3, V3);

    // MUL_
    auto A4 = arange(12).reshape(3, 4);
    auto B4 = ones({3, 4}) * 4;
    A4.Mul_(B4);
    ValuesList V4{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44};
    CheckTensStruct(A4, Type.Double, Device.cpu, {3, 4});
    tens2test(A4, V4);
  }

  TEST(Tensor, equal) {
    // ==
    auto A1 = arange(6).reshape({2, 3});
    auto B1 = arange(6).reshape({2, 3});
    auto C1 = A1 + 1;
    auto cprT = A1 == B1;
    auto cprF = A1 == C1;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        EXPECT_TRUE(cprT(i, j).item<bool>());
        EXPECT_FALSE(cprF(i, j).item<bool>());
      }
    }
  }

  TEST(Tensor, cpr) {
    // cpr
    auto A = arange(6).reshape({2, 3});
    auto B = arange(6).reshape({2, 3});
    auto C = A + 1;
    auto cprT = A.Cpr(B);
    auto cprF = A.Cpr(C);
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        EXPECT_TRUE(cprT(i, j).item<bool>());
        EXPECT_FALSE(cprF(i, j).item<bool>());
      }
    }
  }

  TEST(Tensor, append) {
    // rank 1
    auto A1 = ones({2});
    A1.append(4);
    A1.reshape_({1, 3});
    ValuesList V1{1, 1, 4};
    CheckTensStruct(A1, Type.Double, Device.cpu, {1, 3});
    tens2test(A1, V1);

    // multi-rank
    auto A2 = ones({3, 2});
    auto B2 = ones({2}) * 3;
    ValuesList V2{1, 1, 1, 1, 1, 1, 3, 3};
    A2.append(B2);
    CheckTensStruct(A2, Type.Double, Device.cpu, {4, 2});
    tens2test(A2, V2);
  }

  TEST(Tensor, rank) {
    // rank 2 tensor
    for (auto& assign_device : device_list) {
      for (auto& assign_dtype : dtype_list) {
        Tensor T1 = ones({2, 2}, assign_dtype, assign_device);
        EXPECT_EQ(T1.rank(), 2);
      }
    }

    // rank 3 tensor
    for (auto& assign_device : device_list) {
      for (auto& assign_dtype : dtype_list) {
        Tensor T2 = ones({3, 3}, assign_dtype, assign_device);
        EXPECT_EQ(T2.rank(), 2);
      }
    }
  }
}  // namespace MyTensorTest
