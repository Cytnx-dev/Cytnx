#include "../test_tools.h"
#include "stack_test.h"

using namespace cytnx;
using namespace testing;
using namespace TestTools;

namespace HstackTest {

void CheckResult(const Tensor& hstack_tens, const std::vector<Tensor> &input_tens);
void ErrorTestExcute(const std::vector<Tensor>& Ts);

/*=====test info=====
describe:Input only one tensor. Test all possible data type on cpu device.
input:One tensor with shape = (4, 3), all possible type on cpu device.
====================*/
TEST(Hstack, only_one_tensor) {
  for (auto dtype : dtype_list) {
    if(dtype == Type.Bool) //if both bool type, it will throw error
      continue;
    std::vector<Tensor> Ts = {Tensor({4, 3}, dtype)};
    InitTensorUniform(Ts);
    Tensor hstack_tens = algo::Hstack(Ts);
    CheckResult(hstack_tens, Ts);
  }
}

/*=====test info=====
describe:Input multiple tensor with same type. Test all possible data type on cpu device.
input:Three tensor with the shapes [(4, 3), (4, 2), (4, 5)] and same data type. Test all possible type on cpu device.
====================*/
TEST(Hstack, multi_tensor) {
  for (auto dtype : dtype_list) {
    if(dtype == Type.Bool) //if both bool type, it will throw error
      continue;
    std::vector<Tensor> Ts = {
        Tensor({4, 3}, dtype), 
        Tensor({4, 2}, dtype), 
        Tensor({4, 5}, dtype)
    };
    InitTensorUniform(Ts);
    Tensor hstack_tens = algo::Hstack(Ts);
    CheckResult(hstack_tens, Ts);
  }
}

/*=====test info=====
describe:Test arbitray two data type.
input:Three tensor on cpu device with the shapes [(4, 3), (4, 2)] but the data type may be different. Test all possible data type with arbitrary combination.
====================*/
TEST(Hstack, two_type_tensor) {
  for (auto dtype1 : dtype_list) {
    for (auto dtype2 : dtype_list) {
      if(dtype1 == Type.Bool && dtype2 ==Type.Bool) //if both are bool type, it will throw error.
        continue;
      std::vector<Tensor> Ts = {
          Tensor({4, 3}, dtype1), 
          Tensor({4, 2}, dtype2)
      };
      InitTensorUniform(Ts);
      Tensor hstack_tens = algo::Hstack(Ts);
      CheckResult(hstack_tens, Ts);
    }
  }
}

/*=====test info=====
describe:Test multiple tensor with different data type.
input:Four tensor as following
  T1:shape (4, 2) with bool type on cpu device.
  T2:shape (4, 3) with complex double type on cpu device.
  T3:shape (4, 2) with double type on cpu device.
  T4:shape (4, 1) with double type on cpu device.
  T5:shape (4, 5) with uint64 type on cpu device.
====================*/
TEST(Hstack, diff_type_tensor) {
  std::vector<Tensor> Ts = {
      Tensor({4, 2}, Type.Bool), 
      Tensor({4, 3}, Type.ComplexDouble), 
      Tensor({4, 2}, Type.Double), 
      Tensor({4, 1}, Type.Double), 
      Tensor({4, 5}, Type.Uint64)
  };
  InitTensorUniform(Ts);
  Tensor hstack_tens = algo::Hstack(Ts);
  CheckResult(hstack_tens, Ts);
}

//error test

/*=====test info=====
describe:Test input empty vector.
input:empty vector, cpu
====================*/
TEST(Hstack, err_empty) {
  std::vector<Tensor> empty;
  ErrorTestExcute(empty);
}

/*=====test info=====
describe:Test input is void tensor
input:void tensor, cpu
====================*/
TEST(Hstack, err_tensor_void) {
  std::vector<Tensor> Ts = {Tensor()};
  InitTensorUniform(Ts);
  ErrorTestExcute(Ts);
}

/*=====test info=====
describe:Test input contains void tensor
input:
  T1:shape (4, 3), data type is double, cpu
  T2:void tensor
====================*/
TEST(Hstack, err_contains_void) {
  std::vector<Tensor> Ts = {
      Tensor({4, 3}, Type.Double), 
      Tensor()
  };
  InitTensorUniform(Ts);
  ErrorTestExcute(Ts);
}

/*=====test info=====
describe:Test input different row matrix.
input:
  T1:shape (4, 3), data type is double, cpu
  T2:shape (2, 3), data type is double, cpu
====================*/
TEST(Hstack, err_row_not_eq) {
  std::vector<Tensor> Ts = {
      Tensor({4, 3}, Type.Double), 
      Tensor({2, 3}, Type.Double)
  };
  InitTensorUniform(Ts);
  ErrorTestExcute(Ts);
}

/*=====test info=====
describe:Test input bool type.
input:
  T:shape (2, 3), data type is bool, cpu
====================*/
TEST(Hstack, err_a_bool_type) {
  std::vector<Tensor> Ts = {Tensor({2, 3}, Type.Bool)};
  InitTensorUniform(Ts);
  ErrorTestExcute(Ts);
}

/*=====test info=====
describe:Test input types are all bool.
input:
  T1:shape (4, 3), data type is bool, cpu
  T1:shape (4, 2), data type is bool, cpu
====================*/
TEST(Hstack, err_multi_bool_type) {
  std::vector<Tensor> Ts = {
      Tensor({4, 3}, Type.Bool), 
      Tensor({4, 2}, Type.Bool)
  };
  InitTensorUniform(Ts);
  ErrorTestExcute(Ts);
}

void InitTensorUniform(std::vector<Tensor>& Ts) {
  unsigned int rand_seed = 0;
  for(auto& T : Ts) {
    auto dtype = T.dtype();
    EXPECT_EQ(T.device(), Device.cpu); //for cuda still not implement
    if(dtype == Type.Void) 
      continue;
    //The function 'astype' still not implement for casting complex to real currently.
    //  if 'astype' implement cast from comlex to double, we can just cast from complex to another.
    auto tmp_type = (dtype == Type.ComplexDouble || dtype == Type.ComplexFloat) ? 
        Type.ComplexDouble : Type.Double;
    Tensor tmp = Tensor(T.shape(), tmp_type, T.device());
    rand_seed++;
    double l_bd;
    double h_bd;
    switch (dtype) {
      case Type.Void: //never
        continue;
      case Type.ComplexDouble:
      case Type.ComplexFloat:
      case Type.Double:
      case Type.Float:
      case Type.Int64:
      case Type.Int32: 
        l_bd = -1000000, h_bd = 1000000;
        break;
      case Type.Uint64:
      case Type.Uint32:
        l_bd = 0, h_bd = 1000000;
        break;
      case Type.Int16: 
	l_bd = std::numeric_limits<int16_t>::min();
	h_bd = std::numeric_limits<int16_t>::max();
        break;
      case Type.Uint16: 
	l_bd = std::numeric_limits<uint16_t>::min();
	h_bd = std::numeric_limits<uint16_t>::max();
        break;
      case Type.Bool: 
        l_bd = 0.0, h_bd = 2.0;
        break;
      default: //wrong input 
        FAIL();
    } //switch
    random::Make_uniform(tmp, l_bd, h_bd, rand_seed);
    if(dtype == Type.Bool) {
      //bool type prepare:double in range (0, 2) -> uint32 [0, 1] ->bool
      //  bool type prepare:1.X -> 1 ->true; 0.X -> 0 ->false
      tmp = tmp.astype(Type.Uint32); 
    }
    T = tmp.astype(dtype);
  } //for
} // func:InitTensorUniform

bool IsElemSame(const Tensor& T1, const std::vector<cytnx_uint64>& idices1,
                const Tensor& T2, const std::vector<cytnx_uint64>& idices2) {
  if(T1.dtype() != T2.dtype())
    return false;
  if(T1.device() != T2.device())
    return false;
  switch (T1.dtype()) {
    case Type.Void:
      break;
    case Type.ComplexDouble: {
      auto t1_val = T1.at<std::complex<double>>(idices1);
      auto t2_val = T2.at<std::complex<double>>(idices2);
      if(t1_val != t2_val)
        return false;
      break;
    }
    case Type.ComplexFloat: {
      auto t1_val = T1.at<std::complex<float>>(idices1);
      auto t2_val = T2.at<std::complex<float>>(idices2);
      if(t1_val != t2_val)
        return false;
      break;
    }
    case Type.Double: {
      auto t1_val = T1.at<double>(idices1);
      auto t2_val = T2.at<double>(idices2);
      if(t1_val != t2_val)
        return false;
      break;
    }
    case Type.Float: {
      auto t1_val = T1.at<float>(idices1);
      auto t2_val = T2.at<float>(idices2);
      if(t1_val != t2_val)
        return false;
      break;
    }
    case Type.Int64: {
      auto t1_val = T1.at<int64_t>(idices1);
      auto t2_val = T2.at<int64_t>(idices2);
      if(t1_val != t2_val)
        return false;
      break;
    }
    case Type.Uint64: {
      auto t1_val = T1.at<uint64_t>(idices1);
      auto t2_val = T2.at<uint64_t>(idices2);
      if(t1_val != t2_val)
        return false;
      break;
    }
    case Type.Int32: {
      auto t1_val = T1.at<int32_t>(idices1);
      auto t2_val = T2.at<int32_t>(idices2);
      if(t1_val != t2_val)
        return false;
      break;
    }
    case Type.Uint32: {
      auto t1_val = T1.at<uint32_t>(idices1);
      auto t2_val = T2.at<uint32_t>(idices2);
      if(t1_val != t2_val)
        return false;
      break;
    }
    case Type.Int16: {
      auto t1_val = T1.at<int16_t>(idices1);
      auto t2_val = T2.at<int16_t>(idices2);
      if(t1_val != t2_val)
        return false;
      break;
    }
    case Type.Uint16: {
      auto t1_val = T1.at<uint16_t>(idices1);
      auto t2_val = T2.at<uint16_t>(idices2);
      if(t1_val != t2_val)
        return false;
      break;
    }
    case Type.Bool: {
      auto t1_val = T1.at<bool>(idices1);
      auto t2_val = T2.at<bool>(idices2);
      if(t1_val != t2_val)
        return false;
      break;
    }
    default:
      return false;
  } //switch
  return true;
} //func:CheckElemSame

void CheckResult(const Tensor& hstack_tens, 
		 const std::vector<Tensor> &input_tens) {
  //1. check tensor data type
  std::vector<unsigned int> input_types;
  for(size_t i = 0; i < input_tens.size(); ++i) {
    input_types.push_back(input_tens[i].dtype());
  }
  //since complex double < complex float < double < ... < bool, we need to 
  //  find the min value type as the final converted type.
  auto expect_dtype = *std::min_element(input_types.begin(), input_types.end());
  EXPECT_EQ(expect_dtype, hstack_tens.dtype());

  //2. check tensor shape
  auto hstack_shape = hstack_tens.shape();
  EXPECT_EQ(hstack_shape.size(), 2); //need to be matrix
  int D_share = input_tens[0].shape()[0];
  int D_total = 0;
  for(auto tens : input_tens) {
    ASSERT_EQ(tens.shape()[0], D_share); //all column need to be same
    D_total += tens.shape()[1];
  }
  EXPECT_EQ(hstack_shape[0], D_share);
  EXPECT_EQ(hstack_shape[1], D_total);

  //3. check tensor elements
  int block_col_shift = 0;
  bool is_same_elem = true;
  for(auto tens : input_tens) {
    auto cvt_tens = tens.astype(expect_dtype);
    auto src_shape = tens.shape();
    auto r_num = src_shape[0]; //row number 
    auto c_num = src_shape[1]; //column number
    for(int r = 0; r < r_num; ++r) {
      for(int c = 0; c < c_num; ++c) {
        auto dst_c = c + block_col_shift;
	is_same_elem = IsElemSame(cvt_tens, {r, c}, hstack_tens, {r, dst_c});
	if(!is_same_elem)
          break;
      } //end col
    } //end row
    block_col_shift += c_num;
  } //end input tens vec
  EXPECT_TRUE(is_same_elem);
} //fucn:CheckResult

void ErrorTestExcute(const std::vector<Tensor>& Ts) {
  try {
    Tensor tens = algo::Hstack(Ts);
    std::cerr << "[Test Error] This test should throw error but not !" << std::endl;
    FAIL();
  } catch(const std::exception& ex) {
    auto err_msg = ex.what();
    std::cerr << err_msg << std::endl;
    SUCCEED();
  }
}

} //namespace
