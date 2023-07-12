#include "test_tools.h"

#define RAND_MAX_VAL 1000
#define RAND_MIN_VAL ((-1) * RAND_MAX_VAL)

using namespace cytnx;
using namespace testing;

namespace TestTools {

  void TestFailMsg::AppendMsg(const std::string& fail_msg, const std::string& func_name,
                              const int line) {
    std::string full_msg;
    full_msg += fail_msg;
    full_msg += ("Please check the function " + func_name + " and see the line " +
                 std::to_string(line) + ".\n");
    AppendMsg(full_msg);
  }

  std::string TestFailMsg::TraceFailMsgs() {
    std::string trace_msg;
    trace_msg = "Test case: " + test_case + "\n Trace message:\n";
    std::string spaces;
    while (!fail_msgs.empty()) {
      spaces += "  ";
      trace_msg += (spaces + fail_msgs.back());
      fail_msgs.pop_back();
    }
    return trace_msg;
  }

  bool AreNearlyEqStorage(const Storage& stor1, const Storage& stor2, const cytnx_double tol) {
    if (tol == 0) {
      return (const_cast<Storage&>(stor1) == const_cast<Storage&>(stor2));
    } else {
      if (stor1.dtype() != stor2.dtype()) {
        std::cout << "[AreNearlyEqStorage] storage1 type: " << stor1.dtype() << " does not match "
                  << stor2.dtype() << std::endl;
        return false;
      }
      if (stor1.size() != stor2.size()) {
        std::cout << "[AreNearlyEqStorage] storage1 size: " << stor1.size() << " does not match "
                  << stor2.size() << std::endl;
        return false;
      }
      auto dtype = stor1.dtype();
      auto size = stor1.size();

      switch (dtype) {
        case Type.ComplexDouble:
          for (cytnx_uint64 i = 0; i < size; i++) {
            if (abs(stor1.at<cytnx_complex128>(i) - stor2.at<cytnx_complex128>(i)) > tol) {
              std::cout << "[AreNearlyEqStorage] element" << i << " : "
                        << stor1.at<cytnx_complex128>(i) << " != " << stor2.at<cytnx_complex128>(i)
                        << std::endl;
              return false;
            }
          }
          break;
        case Type.ComplexFloat:
          for (cytnx_uint64 i = 0; i < size; i++) {
            if (abs(stor1.at<cytnx_complex64>(i) - stor2.at<cytnx_complex64>(i)) > tol) {
              std::cout << "[AreNearlyEqStorage] element" << i << " : "
                        << stor1.at<cytnx_complex64>(i) << " != " << stor2.at<cytnx_complex64>(i)
                        << std::endl;
              return false;
            }
          }
          break;
        case Type.Double:
          for (cytnx_uint64 i = 0; i < size; i++) {
            if (abs(stor1.at<cytnx_double>(i) - stor2.at<cytnx_double>(i)) > tol) {
              std::cout << "[AreNearlyEqStorage] element" << i << " : " << stor1.at<cytnx_double>(i)
                        << " != " << stor2.at<cytnx_double>(i) << std::endl;
              return false;
            }
          }
          break;
        case Type.Float:
          for (cytnx_uint64 i = 0; i < size; i++) {
            if (abs(stor1.at<cytnx_float>(i) - stor2.at<cytnx_float>(i)) > tol) {
              std::cout << "[AreNearlyEqStorage] element" << i << " : " << stor1.at<cytnx_float>(i)
                        << " != " << stor2.at<cytnx_float>(i) << std::endl;
              return false;
            }
          }
          break;
        case Type.Int64:
          for (cytnx_uint64 i = 0; i < size; i++) {
            if (abs(stor1.at<cytnx_int64>(i) - stor2.at<cytnx_int64>(i)) > tol) {
              std::cout << "[AreNearlyEqStorage] element" << i << " : " << stor1.at<cytnx_int64>(i)
                        << " != " << stor2.at<cytnx_int64>(i) << std::endl;
              return false;
            }
          }
          break;
        case Type.Uint64:
          for (cytnx_uint64 i = 0; i < size; i++) {
            if (abs(static_cast<double>(stor1.at<cytnx_uint64>(i)) -
                    static_cast<double>(stor2.at<cytnx_uint64>(i))) > tol) {
              std::cout << "[AreNearlyEqStorage] element" << i << " : " << stor1.at<cytnx_uint64>(i)
                        << " != " << stor2.at<cytnx_uint64>(i) << std::endl;
              return false;
            }
          }
          break;
        case Type.Int32:
          for (cytnx_uint64 i = 0; i < size; i++) {
            if (abs(stor1.at<cytnx_int32>(i) - stor2.at<cytnx_int32>(i)) > tol) {
              std::cout << "[AreNearlyEqStorage] element" << i << " : " << stor1.at<cytnx_int32>(i)
                        << " != " << stor2.at<cytnx_int32>(i) << std::endl;
              return false;
            }
          }
          break;
        case Type.Uint32:
          for (cytnx_uint64 i = 0; i < size; i++) {
            if (abs(static_cast<double>(stor1.at<cytnx_uint32>(i)) -
                    static_cast<double>(stor2.at<cytnx_uint32>(i))) > tol) {
              std::cout << "[AreNearlyEqStorage] element" << i << " : " << stor1.at<cytnx_uint32>(i)
                        << " != " << stor2.at<cytnx_uint32>(i) << std::endl;
              return false;
            }
          }
          break;
        case Type.Int16:
          for (cytnx_uint64 i = 0; i < size; i++) {
            if (abs(stor1.at<cytnx_int16>(i) - stor2.at<cytnx_int16>(i)) > tol) {
              std::cout << "[AreNearlyEqStorage] element" << i << " : " << stor1.at<cytnx_int16>(i)
                        << " != " << stor2.at<cytnx_int16>(i) << std::endl;
              return false;
            }
          }
          break;
        case Type.Uint16:
          for (cytnx_uint64 i = 0; i < size; i++) {
            if (abs(static_cast<double>(stor1.at<cytnx_uint16>(i)) -
                    static_cast<double>(stor2.at<cytnx_uint16>(i))) > tol) {
              std::cout << "[AreNearlyEqStorage] element" << i << " : " << stor1.at<cytnx_uint16>(i)
                        << " != " << stor2.at<cytnx_uint16>(i) << std::endl;
              return false;
            }
          }
          break;
        case Type.Bool:
          for (cytnx_uint64 i = 0; i < size; i++) {
            if (abs(static_cast<double>(stor1.at<cytnx_bool>(i)) -
                    static_cast<double>(stor2.at<cytnx_bool>(i))) > tol) {
              std::cout << "[AreNearlyEqStorage] element" << i << " : " << stor1.at<cytnx_bool>(i)
                        << " != " << stor2.at<cytnx_bool>(i) << std::endl;
              return false;
            }
          }
          break;
        default:
          cytnx_error_msg(true, "[ERROR] fatal internal, Storage has invalid type.%s", "\n");
          return false;
      }  // switch
      return true;
    }  // else
  }

  // Tensor
  // random initialize
  void GetRandRange(const unsigned int dtype, cytnx_double* low_bd, cytnx_double* high_bd) {
    if (dtype == Type.Void) return;
    switch (dtype) {
      case Type.Void:  // return directly
        return;
      case Type.ComplexDouble:
      case Type.ComplexFloat:
      case Type.Double:
      case Type.Float:
      case Type.Int64:
      case Type.Int32:
        *low_bd = RAND_MIN_VAL, *high_bd = RAND_MAX_VAL;
        break;
      case Type.Uint64:
      case Type.Uint32:
        *low_bd = 0, *high_bd = RAND_MAX_VAL;
        break;
      case Type.Int16:
        *low_bd = std::numeric_limits<int16_t>::min();
        *high_bd = std::numeric_limits<int16_t>::max();
        break;
      case Type.Uint16:
        *low_bd = std::numeric_limits<uint16_t>::min();
        *high_bd = std::numeric_limits<uint16_t>::max();
        break;
      case Type.Bool:
        *low_bd = 0.0, *high_bd = 2.0;
        break;
      default:  // wrong input
        break;
    }  // switch
  }

  void InitTensorUniform(Tensor& T, unsigned int rand_seed) {
    auto dtype = T.dtype();
    if (dtype == Type.Void) return;
    cytnx_double l_bd, h_bd;
    GetRandRange(dtype, &l_bd, &h_bd);
    //  if 'astype' implement cast from comlex to double, we can just cast from complex to another.
    auto tmp_type = (dtype == Type.ComplexDouble || dtype == Type.ComplexFloat) ? Type.ComplexDouble
                                                                                : Type.Double;
    Tensor tmp = Tensor(T.shape(), tmp_type, T.device());
    random::Make_uniform(tmp, l_bd, h_bd, rand_seed);
    if (dtype == Type.Bool) {
      // bool type prepare:double in range (0, 2) -> uint32 [0, 1] ->bool
      //   bool type prepare:1.X -> 1 ->true; 0.X -> 0 ->false
      tmp = tmp.astype(Type.Uint32);
    }
    T = tmp.astype(dtype);
  }  // func:InitTensUniform

  void InitTensorUniform(std::vector<Tensor>& Ts, unsigned int rand_seed) {
    for (auto& T : Ts) {
      InitTensorUniform(T, rand_seed++);
    }
  }

  // comparison
  bool AreNearlyEqTensor(const Tensor& T1, const Tensor& T2, const cytnx_double tol) {
    if (T1.device() != T2.device()) {
      std::cout << "[AreNearlyEqTensor] T1 device " << T1.device() << "not equal to T2 device "
                << T2.device() << std::endl;
      return false;
    }
    if (T1.dtype() != T2.dtype()) {
      std::cout << "[AreNearlyEqTensor] T1 dtype " << T1.dtype() << "not equal to T2 dtype "
                << T2.dtype() << std::endl;
      return false;
    }
    if (T1.shape() != T2.shape()) {
      std::cout << "[AreNearlyEqTensor] T1 shape " << T1.shape() << "not equal to T2 shape "
                << T2.shape() << std::endl;
      return false;
    }
    if (T1.is_contiguous() != T2.is_contiguous()) {
      std::cout << "[AreNearlyEqTensor] T1 contiguous flag " << T1.is_contiguous()
                << "not equal to T2 flag " << T2.is_contiguous() << std::endl;
      return false;
    }
    return AreNearlyEqStorage(T1.storage(), T2.storage(), tol);
  }

  bool AreEqTensor(const Tensor& T1, const Tensor& T2) {
    const double tol = 0;
    return AreNearlyEqTensor(T1, T2, tol);
  }

  bool AreElemSame(const Tensor& T1, const std::vector<cytnx_uint64>& idices1, const Tensor& T2,
                   const std::vector<cytnx_uint64>& idices2) {
    if (T1.dtype() != T2.dtype()) {
      std::cout << "[AreElemSame] T1 dtype " << T1.dtype() << "not equal to T2 dtype " << T2.dtype()
                << std::endl;
      return false;
    }
    if (T1.device() != T2.device()) {
      std::cout << "[AreElemSame] T1 device " << T1.device() << "not equal to T2 device "
                << T2.device() << std::endl;
      return false;
    }
    // we don't need to check tensor shape here because we want to compare the
    //   different shape result elements.
    try {
      switch (T1.dtype()) {
        case Type.Void:
          break;
        case Type.ComplexDouble: {
          auto t1_val = T1.at<std::complex<double>>(idices1);
          auto t2_val = T2.at<std::complex<double>>(idices2);
          if (t1_val != t2_val) {
            std::cout << "[AreElemSame]"
                      << "T1[" << idices1 << "] = " << t1_val << ", "
                      << "T2[" << idices2 << "] = " << t2_val << ", not same." << std::endl;
            return false;
          }
          break;
        }
        case Type.ComplexFloat: {
          auto t1_val = T1.at<std::complex<float>>(idices1);
          auto t2_val = T2.at<std::complex<float>>(idices2);
          if (t1_val != t2_val) {
            std::cout << "[AreElemSame]"
                      << "T1[" << idices1 << "] = " << t1_val << ", "
                      << "T2[" << idices2 << "] = " << t2_val << ", not same." << std::endl;
            return false;
          }
          break;
        }
        case Type.Double: {
          auto t1_val = T1.at<double>(idices1);
          auto t2_val = T2.at<double>(idices2);
          if (t1_val != t2_val) {
            std::cout << "[AreElemSame]"
                      << "T1[" << idices1 << "] = " << t1_val << ", "
                      << "T2[" << idices2 << "] = " << t2_val << ", not same." << std::endl;
            return false;
          }
          break;
        }
        case Type.Float: {
          auto t1_val = T1.at<float>(idices1);
          auto t2_val = T2.at<float>(idices2);
          if (t1_val != t2_val) {
            std::cout << "[AreElemSame]"
                      << "T1[" << idices1 << "] = " << t1_val << ", "
                      << "T2[" << idices2 << "] = " << t2_val << ", not same." << std::endl;
            return false;
          }
          break;
        }
        case Type.Int64: {
          auto t1_val = T1.at<int64_t>(idices1);
          auto t2_val = T2.at<int64_t>(idices2);
          if (t1_val != t2_val) {
            std::cout << "[AreElemSame]"
                      << "T1[" << idices1 << "] = " << t1_val << ", "
                      << "T2[" << idices2 << "] = " << t2_val << ", not same." << std::endl;
            return false;
          }
          break;
        }
        case Type.Uint64: {
          auto t1_val = T1.at<uint64_t>(idices1);
          auto t2_val = T2.at<uint64_t>(idices2);
          if (t1_val != t2_val) {
            std::cout << "[AreElemSame]"
                      << "T1[" << idices1 << "] = " << t1_val << ", "
                      << "T2[" << idices2 << "] = " << t2_val << ", not same." << std::endl;
            return false;
          }
          break;
        }
        case Type.Int32: {
          auto t1_val = T1.at<int32_t>(idices1);
          auto t2_val = T2.at<int32_t>(idices2);
          if (t1_val != t2_val) {
            std::cout << "[AreElemSame]"
                      << "T1[" << idices1 << "] = " << t1_val << ", "
                      << "T2[" << idices2 << "] = " << t2_val << ", not same." << std::endl;
            return false;
          }
          break;
        }
        case Type.Uint32: {
          auto t1_val = T1.at<uint32_t>(idices1);
          auto t2_val = T2.at<uint32_t>(idices2);
          if (t1_val != t2_val) {
            std::cout << "[AreElemSame]"
                      << "T1[" << idices1 << "] = " << t1_val << ", "
                      << "T2[" << idices2 << "] = " << t2_val << ", not same." << std::endl;
            return false;
          }
          break;
        }
        case Type.Int16: {
          auto t1_val = T1.at<int16_t>(idices1);
          auto t2_val = T2.at<int16_t>(idices2);
          if (t1_val != t2_val) return false;
          break;
        }
        case Type.Uint16: {
          auto t1_val = T1.at<uint16_t>(idices1);
          auto t2_val = T2.at<uint16_t>(idices2);
          if (t1_val != t2_val) {
            std::cout << "[AreElemSame]"
                      << "T1[" << idices1 << "] = " << t1_val << ", "
                      << "T2[" << idices2 << "] = " << t2_val << ", not same." << std::endl;
            return false;
          }
          break;
        }
        case Type.Bool: {
          auto t1_val = T1.at<bool>(idices1);
          auto t2_val = T2.at<bool>(idices2);
          if (t1_val != t2_val) {
            std::cout << "[AreElemSame]"
                      << "T1[" << idices1 << "] = " << t1_val << ", "
                      << "T2[" << idices2 << "] = " << t2_val << ", not same." << std::endl;
            return false;
          }
          break;
        }
        default:
          std::cout << "[AreElemSame] wrong input type" << std::endl;
          return false;
      }  // switch
    }  // try
    catch (const std::exception& ex) {
      cytnx_error_msg(true, "[ERROR][test_tools]", __FUNCTION__, ex.what(), "\n");
    }
    return true;
  }  // func:CheckElemSame

  // UniTensor
  bool AreNearlyEqUniTensor(const UniTensor& Ut1, const UniTensor& Ut2, const cytnx_double tol) {
    if (Ut1.uten_type() != Ut2.uten_type()) {
      std::cout << "[AreNearlyEqUniTensor] uten_type not the same. " << Ut1.uten_type() << " "
                << Ut2.uten_type() << std::endl;
      return false;
    }
    // dense
    if (Ut1.uten_type() == UTenType.Dense) {
      Tensor block1 = Ut1.get_block();
      Tensor block2 = Ut2.get_block();
      if (!AreNearlyEqTensor(block1, block2, tol)) {
        std::cout << "[AreNearlyEqUniTensor] two block not the same. " << std::endl;
        return false;
      }
    }
    // block
    else if (Ut1.uten_type() == UTenType.Block) {
      const std::vector<Tensor>& blocks1 = Ut1.get_blocks_();
      const std::vector<Tensor>& blocks2 = Ut2.get_blocks_();
      if (blocks1.size() != blocks2.size()) {
        std::cout << "# of blocks are not the same btwn two UTs. " << std::endl;
        return false;
      }
      auto blocks_num = blocks1.size();
      for (size_t i = 0; i < blocks_num; ++i) {
        if (!AreNearlyEqTensor(blocks1[i], blocks2[i], tol)) {
          std::cout << "on " << i << " blk, the tensor does not mtach!" << std::endl;
          std::cout << blocks1[i] << std::endl;
          std::cout << blocks2[i] << std::endl;
          return false;
        }
      }
    } else {  // void
      return false;
    }
    return true;
  }

  bool AreEqUniTensor(const UniTensor& Ut1, const UniTensor& Ut2) {
    return AreNearlyEqUniTensor(Ut1, Ut2, 0);
  }

  // UniTensor
  void InitUniTensorUniform(UniTensor& UT, unsigned int rand_seed) {
    auto dtype = UT.dtype();
    if (dtype == Type.Void) return;
    cytnx_double l_bd, h_bd;
    GetRandRange(dtype, &l_bd, &h_bd);
    //  if 'astype' implement cast from comlex to double, we can just cast from complex to another.
    auto tmp_type = (dtype == Type.ComplexDouble || dtype == Type.ComplexFloat) ? Type.ComplexDouble
                                                                                : Type.Double;
    UniTensor tmp = UT.astype(tmp_type);
    random::Make_uniform(tmp, l_bd, h_bd, rand_seed);
    if (dtype == Type.Bool) {
      // bool type prepare:double in range (0, 2) -> uint32 [0, 1] ->bool
      //   bool type prepare:1.X -> 1 ->true; 0.X -> 0 ->false
      tmp = tmp.astype(Type.Uint32);
    }
    UT = tmp.astype(dtype);
  }  // func:InitTensUniform

}  // namespace TestTools
