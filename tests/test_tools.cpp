#include "test_tools.h"

#define RAND_MAX_VAL 1000000
#define RAND_MIN_VAL ((-1) * RAND_MAX_VAL)

using namespace cytnx;
using namespace testing;

namespace TestTools {


bool AreNearlyEqStorage(const Storage& stor1, const Storage& stor2, 
                       const cytnx_double tol) {
  if (tol == 0) {
    return (const_cast<Storage&>(stor1) == const_cast<Storage&>(stor2));
  } else {
    if (stor1.dtype() != stor2.dtype()) 
      return false;
    if (stor1.size() != stor2.size())
      return false;
    auto dtype = stor1.dtype();
    auto size = stor1.size();

    switch (dtype) {
      case Type.ComplexDouble:
        for (cytnx_uint64 i = 0; i < size; i++) {
          if (abs(stor1.at<cytnx_complex128>(i) - stor2.at<cytnx_complex128>(i)) > tol) 
            return false;
        }
        break;
      case Type.ComplexFloat:
        for (cytnx_uint64 i = 0; i < size; i++) {
          if (abs(stor1.at<cytnx_complex64>(i) - stor2.at<cytnx_complex64>(i)) > tol) 
            return false;
        }
        break;
      case Type.Double:
        for (cytnx_uint64 i = 0; i < size; i++) {
          if (abs(stor1.at<cytnx_double>(i) - stor2.at<cytnx_double>(i)) > tol) 
            return false;
        }
        break;
      case Type.Float:
        for (cytnx_uint64 i = 0; i < size; i++) {
          if (abs(stor1.at<cytnx_float>(i) - stor2.at<cytnx_float>(i)) > tol) 
            return false;
        }
        break;
      case Type.Int64:
        for (cytnx_uint64 i = 0; i < size; i++) {
          if (abs(stor1.at<cytnx_int64>(i) - stor2.at<cytnx_int64>(i)) > tol) 
            return false;
        }
        break;
      case Type.Uint64:
        for (cytnx_uint64 i = 0; i < size; i++) {
          if (abs(static_cast<double>(stor1.at<cytnx_uint64>(i)) - 
                  static_cast<double>(stor2.at<cytnx_uint64>(i))) > tol) 
            return false;
        }
        break;
      case Type.Int32:
        for (cytnx_uint64 i = 0; i < size; i++) {
          if (abs(stor1.at<cytnx_int32>(i) - stor2.at<cytnx_int32>(i)) > tol) 
            return false;
        }
        break;
      case Type.Uint32:
        for (cytnx_uint64 i = 0; i < size; i++) {
          if (abs(static_cast<double>(stor1.at<cytnx_uint32>(i)) - 
                  static_cast<double>(stor2.at<cytnx_uint32>(i))) > tol) 
            return false;
        }
        break;
      case Type.Int16:
        for (cytnx_uint64 i = 0; i < size; i++) {
          if (abs(stor1.at<cytnx_int16>(i) - stor2.at<cytnx_int16>(i)) > tol) 
            return false;
        }
        break;
      case Type.Uint16:
        for (cytnx_uint64 i = 0; i < size; i++) {
          if (abs(static_cast<double>(stor1.at<cytnx_uint16>(i)) - 
                  static_cast<double>(stor2.at<cytnx_uint16>(i))) > tol) 
            return false;
        }
        break;
      case Type.Bool:
        for (cytnx_uint64 i = 0; i < size; i++) {
          if (abs(static_cast<double>(stor1.at<cytnx_bool>(i)) - 
                  static_cast<double>(stor2.at<cytnx_bool>(i))) > tol) 
            return false;
        }
        break;
      default:
        cytnx_error_msg(true, "[ERROR] fatal internal, Storage has invalid type.%s", "\n");
	return false;
    }//switch
    return true;
  }//else
}

//Tensor
//random initialize
void InitTensorUniform(Tensor& T, unsigned int rand_seed) {
  auto dtype = T.dtype();
  if (dtype == Type.Void)
    return;
  //  if 'astype' implement cast from comlex to double, we can just cast from complex to another.
  auto tmp_type = (dtype == Type.ComplexDouble || dtype == Type.ComplexFloat) ? 
      Type.ComplexDouble : Type.Double;
  Tensor tmp = Tensor(T.shape(), tmp_type, T.device());
  double l_bd;
  double h_bd;
  switch (dtype) {
    case Type.Void: //return directly
      return;
    case Type.ComplexDouble:
    case Type.ComplexFloat:
    case Type.Double:
    case Type.Float:
    case Type.Int64:
    case Type.Int32: 
      l_bd = RAND_MIN_VAL, h_bd = RAND_MAX_VAL;
      break;
    case Type.Uint64:
    case Type.Uint32:
      l_bd = 0, h_bd = RAND_MAX_VAL;
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
      break;
  } //switch
  random::Make_uniform(tmp, l_bd, h_bd, rand_seed);
  if(dtype == Type.Bool) {
    //bool type prepare:double in range (0, 2) -> uint32 [0, 1] ->bool
    //  bool type prepare:1.X -> 1 ->true; 0.X -> 0 ->false
    tmp = tmp.astype(Type.Uint32); 
  }
  T = tmp.astype(dtype);
} // func:InitTensUniform

void InitTensorUniform(std::vector<Tensor>& Ts, unsigned int rand_seed) {
  for (auto& T : Ts) {
    InitTensorUniform(T, rand_seed++);
  }
}

//comparison
bool AreNearlyEqTensor(const Tensor& T1, const Tensor& T2, const cytnx_double tol) {
  if (T1.device() != T2.device())
    return false;
  if (T1.dtype() != T2.dtype())
    return false;
  if (T1.shape() != T2.shape())
    return false;
  if (T1.is_contiguous() != T2.is_contiguous())
    return false;

  return AreNearlyEqStorage(T1.storage(), T2.storage(), tol);
}

bool AreEqTensor(const Tensor& T1, const Tensor& T2) {
  const double tol = 0;
  return AreNearlyEqTensor(T1, T2, tol);
}



} //namespace
