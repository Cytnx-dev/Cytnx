#include "linalg/linalg_internal_cpu/Kron_internal.hpp"

#include "utils/complex_arithmetic.hpp"
#include "utils/utils_internal_interface.hpp"
#include <algorithm>
//#include "lapack_wrapper.hpp"
#ifdef UNI_OMP
  #include <omp.h>
#endif

#include <boost/preprocessor/array/data.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/punctuation/paren.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/array/elem.hpp>

#define DTYPES_SIMPLE (11, (cd, cf, d, f, i64, u64, i32, u32, i16, u16, b))
#define DTYPES_FULL                                                                              \
  (11, (cytnx_complex128, cytnx_complex64, cytnx_double, cytnx_float, cytnx_int64, cytnx_uint64, \
        cytnx_int32, cytnx_uint32, cytnx_int16, cytnx_uint32, bool))

#define GEN_FUNC1(z, i, j)                                                                                                                                     \
  void BOOST_PP_CAT(BOOST_PP_CAT(Kron_internal_,BOOST_PP_ARRAY_ELEM(i, DTYPES_SIMPLE)),BOOST_PP_CAT(t,BOOST_PP_ARRAY_ELEM(j, DTYPES_SIMPLE)))BOOST_PP_LPAREN()\
      boost::intrusive_ptr<Storage_base> &out, const boost::intrusive_ptr<Storage_base> &Lin, \
      const boost::intrusive_ptr<Storage_base> &Rin, const std::vector<cytnx_uint64> &shape1,    \
      const std::vector<cytnx_uint64> &shape2) {                                                                                                               \
    Kron_general<BOOST_PP_ARRAY_ELEM(i, DTYPES_FULL), BOOST_PP_ARRAY_ELEM(j, DTYPES_FULL)>(                                                                    \
      out, Lin, Rin, shape1, shape2);                                                                                                                          \
  }
#define GEN_FUNC2(z, j, unused) BOOST_PP_REPEAT(11, GEN_FUNC1, j)

namespace cytnx {

  namespace linalg_internal {

    namespace {
      template <class TL, class TR>
      void Kron_general(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &Lin,
                        const boost::intrusive_ptr<Storage_base> &Rin,
                        const std::vector<cytnx_uint64> &shape1,
                        const std::vector<cytnx_uint64> &shape2) {
        TL *_out = (TL *)out->Mem;
        TR *_out2 = (TR *)out->Mem;
        TL *_Lin = (TL *)Lin->Mem;
        TR *_Rin = (TR *)Rin->Mem;

        cytnx_error_msg(shape1.size() != shape2.size(),
                        "[ERROR][Internal Kron] T1 rank != T2 rank %s", "\n");
        cytnx_uint64 TotalElem = shape1[0] * shape2[0];
        std::vector<cytnx_uint64> new_shape_acc(shape1.size());
        std::vector<cytnx_uint64> shape1_acc(shape1.size());
        std::vector<cytnx_uint64> shape2_acc(shape1.size());
        new_shape_acc.back() = 1;
        shape1_acc.back() = 1;
        shape2_acc.back() = 1;

        for (unsigned long long i = 1; i < new_shape_acc.size(); i++) {
          new_shape_acc[new_shape_acc.size() - 1 - i] = new_shape_acc[new_shape_acc.size() - i] *
                                                        shape1[new_shape_acc.size() - i] *
                                                        shape2[new_shape_acc.size() - i];
          TotalElem *= shape1[i] * shape2[i];
          shape1_acc[shape1_acc.size() - 1 - i] =
            shape1_acc[shape1_acc.size() - i] * shape1[shape1_acc.size() - i];
          shape2_acc[shape2_acc.size() - 1 - i] =
            shape2_acc[shape2_acc.size() - i] * shape2[shape2_acc.size() - i];
        }

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < TotalElem; i++) {
          cytnx_uint64 tmp = i, tmp2;
          cytnx_uint64 x = 0, y = 0;
          for (unsigned long long j = 0; j < new_shape_acc.size(); j++) {
            tmp2 = tmp / new_shape_acc[j];
            tmp %= new_shape_acc[j];
            x += cytnx_uint64(tmp2 / shape2[j]) * shape1_acc[j];
            y += cytnx_uint64(tmp2 % shape2[j]) * shape2_acc[j];
          }
          if constexpr (sizeof(TL) > sizeof(TR) or ((std::is_same<TL, cytnx_complex128>::value or
                                                     std::is_same<TL, cytnx_complex64>::value) and
                                                    !(std::is_same<TR, cytnx_complex128>::value or
                                                      std::is_same<TR, cytnx_complex64>::value))) {
            _out[i] = _Lin[tmp] * _Rin[tmp2];
          } else {
            _out2[i] = _Lin[tmp] * _Rin[tmp2];
          }
        }
      }

    }  // namespace

    BOOST_PP_REPEAT(11, GEN_FUNC2, ~)
  }  // namespace linalg_internal
}  // namespace cytnx

