#include "utils/vec_io.hpp"
namespace cytnx {

  unsigned long long FileSize(const char* sFileName) {
    std::ifstream f;
    f.open(sFileName, std::ios_base::binary | std::ios_base::in);
    if (!f.good() || f.eof() || !f.is_open()) {
      return 0;
    }
    f.seekg(0, std::ios_base::beg);
    std::ifstream::pos_type begin_pos = f.tellg();
    f.seekg(0, std::ios_base::end);
    return static_cast<unsigned long long>(f.tellg() - begin_pos);
  }

  /*
  template std::vector<cytnx_complex128> vec_map(const std::vector<cytnx_complex128> &,const
  std::vector<cytnx_uint64> &); template std::vector<cytnx_complex64> vec_map(const
  std::vector<cytnx_complex64> &,const std::vector<cytnx_uint64> &); template
  std::vector<cytnx_double> vec_map(const std::vector<cytnx_double> &,const
  std::vector<cytnx_uint64> &); template std::vector<cytnx_float> vec_map(const
  std::vector<cytnx_float> &,const std::vector<cytnx_uint64> &); template std::vector<cytnx_int64>
  vec_map(const std::vector<cytnx_int64> &,const std::vector<cytnx_uint64> &); template
  std::vector<cytnx_uint64> vec_map(const std::vector<cytnx_uint64> &,const
  std::vector<cytnx_uint64> &); template std::vector<cytnx_int32> vec_map(const
  std::vector<cytnx_int32> &,const std::vector<cytnx_uint64> &); template std::vector<cytnx_uint32>
  vec_map(const std::vector<cytnx_uint32> &,const std::vector<cytnx_uint64> &); template
  std::vector<cytnx_int16> vec_map(const std::vector<cytnx_int16> &,const std::vector<cytnx_uint64>
  &); template std::vector<cytnx_uint16> vec_map(const std::vector<cytnx_uint16> &,const
  std::vector<cytnx_uint64> &); template std::vector<cytnx_bool> vec_map(const
  std::vector<cytnx_bool> &,const std::vector<cytnx_uint64> &);

  */
}  // namespace cytnx
