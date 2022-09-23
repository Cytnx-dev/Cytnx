#include "linalg.hpp"
#include "utils/utils.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
#include "cytnx.hpp"
#ifdef UNI_OMP
  #include <omp.h>
#endif

using namespace std;
namespace cytnx {
  namespace linalg {
    cytnx::UniTensor Trace(const cytnx::UniTensor &Tin, const cytnx_int64 &a, const cytnx_int64 &b,
                           const bool &by_label) {
      return Tin.Trace(a, b, by_label);
    }
  }  // namespace linalg
}  // namespace cytnx

namespace cytnx {

  template <class T>
  void _trace_2d(Tensor &out, const Tensor &Tn, const cytnx_uint64 &Ndiag) {
    T a = 0;
    T *rawdata = Tn.storage().data<T>();
    cytnx_uint64 Ldim = Tn.shape()[1];
    for (cytnx_uint64 i = 0; i < Ndiag; i++) a += rawdata[i * Ldim + i];
    out.storage().at<T>(0) = a;
  }

  template <class T>
  void _trace_nd(Tensor &out, const Tensor &Tn, const cytnx_uint64 &Ndiag,
                 const cytnx_uint64 &Nelem, const vector<cytnx_uint64> &accu,
                 const vector<cytnx_uint64> &remain_rank_id, const vector<cytnx_int64> &shape,
                 const cytnx_uint64 &ax1, const cytnx_uint64 &ax2) {
    UniTensor I = UniTensor(eye(Ndiag), false, -1);
    I.set_labels({0, 1});
    UniTensor UTn = UniTensor(Tn, false, 2);
    UTn.set_labels(vec_cast<cytnx_uint64, cytnx_int64>(vec_range(100, 100 + UTn.labels().size())));
    UTn.set_label(ax1, 0);
    UTn.set_label(ax2, 1);
    out = Contract(I, UTn).get_block_();

    // vector<cytnx_uint64> indexer(Tn.shape().size(), 0);
    // cytnx_uint64 tmp;
    // for (cytnx_uint64 i = 0; i < Nelem; i++) {
    // tmp = i;
    // // calculate indexer
    // for (int x = 0; x < shape.size(); x++) {
    // indexer[remain_rank_id[x]] = cytnx_uint64(tmp / accu[x]);
    // tmp %= accu[x];
    // }

    // for (cytnx_uint64 d = 0; d < Ndiag; d++) {
    // indexer[ax1] = indexer[ax2] = d;
    // out.storage().at<T>(i) += Tn.at<T>(indexer);
    // }
    // }
  }

#ifdef UNI_OMP
  template <class T>
  void _trace_2d_para(Tensor &out, const Tensor &Tn, const cytnx_uint64 &Ndiag, const int &Nomp) {
    T a = 0;
    vector<T> buffer(Nomp);

  #pragma omp parallel for schedule(dynamic)
    for (cytnx_uint64 i = 0; i < Ndiag; i++) buffer[omp_get_thread_num()] += Tn.at<T>({i, i});

    for (int i = 1; i < Nomp; i++) buffer[0] += buffer[i];
    out.storage().at<T>({0}) = buffer[0];
  }

  template <class T>
  void _trace_nd_para(Tensor &out, const Tensor &Tn, const cytnx_uint64 &Ndiag,
                      const cytnx_uint64 &Nelem, const vector<cytnx_uint64> &accu,
                      const vector<cytnx_uint64> &remain_rank_id, const vector<cytnx_int64> &shape,
                      const cytnx_uint64 &ax1, const cytnx_uint64 &ax2, const int &Nomp) {
    // decide parallel Nelem or Ndiag:
    if (false and Nelem < Ndiag) {
      // each thread need it's own indexer:
      vector<vector<cytnx_uint64>> indexers(Nomp, vector<cytnx_uint64>(Tn.shape().size(), 0));
  // cout << "Ne < Nd" << endl;
  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 i = 0; i < Nelem; i++) {
        cytnx_uint64 tmp = i;
        // calculate indexer
        for (int x = 0; x < shape.size(); x++) {
          indexers[omp_get_thread_num()][remain_rank_id[x]] = cytnx_uint64(tmp / accu[x]);
          tmp %= accu[x];
        }

        for (cytnx_uint64 d = 0; d < Ndiag; d++) {
          indexers[omp_get_thread_num()][ax1] = indexers[omp_get_thread_num()][ax2] = d;
          out.storage().at<T>(i) += Tn.at<T>(indexers[omp_get_thread_num()]);
        }
      }

    } else {
  #pragma omp parallel
      {
        vector<cytnx_uint64> indexers(Tn.shape().size(), 0);
  #pragma omp for schedule(static)
        for (cytnx_uint64 i = 0; i < Nelem; i++) {
          cytnx_uint64 tmp;
          tmp = i;
          // calculate indexer
          for (int x = 0; x < shape.size(); x++) {
            indexers[remain_rank_id[x]] = cytnx_uint64(tmp / accu[x]);
            tmp %= accu[x];
          }

          for (cytnx_uint64 d = 0; d < Ndiag; d++) {
            indexers[ax1] = indexers[ax2] = d;
            out.storage().at<T>(i) += Tn.at<T>(indexers);
          }
        }
      }
    }
  }
#endif

  namespace linalg {
    // dtype -1: default
    // device -2: default.
    Tensor Trace(const Tensor &Tn, const cytnx_uint64 &axisA, const cytnx_uint64 &axisB) {
      // checking:
      cytnx_error_msg(Tn.shape().size() < 2, "[ERROR] Tensor must have at least rank-2.%s", "\n");
      cytnx_error_msg(axisA >= Tn.shape().size(), "[ERROR] axisA out of bound.%s", "\n");
      cytnx_error_msg(axisB >= Tn.shape().size(), "[ERROR] axisB out of bound.%s", "\n");
      cytnx_error_msg(axisA == axisB, "[ERROR] axisB cannot be the same as axisA.%s", "\n");
      // cytnx_error_msg(dtype == Type.Void,"[ERROR] cannot have output type to be
      // Type.Void.%s","\n"); vector<cytnx_int64> indexer(Tn.shape().size());

      cytnx_uint64 ax1, ax2;
      if (axisA < axisB) {
        ax1 = axisA;
        ax2 = axisB;
      } else {
        ax1 = axisB;
        ax2 = axisA;
      }

      // int out_dtype = dtype==-1?Tn.dtype():dtype;
      // int out_device = device==-2?Tn.device():device;

      // 1) get redundant rank:
      vector<cytnx_int64> shape(Tn.shape().begin(), Tn.shape().end());
      vector<cytnx_uint64> accu;
      shape.erase(shape.begin() + ax2);
      shape.erase(shape.begin() + ax1);
      // 2) get out put elementsize.
      cytnx_uint64 Nelem = 1;
      for (int i = 0; i < shape.size(); i++) Nelem *= shape[i];
      // 3) get diagonal element numbers:
      cytnx_uint64 Ndiag = Tn.shape()[ax1] < Tn.shape()[ax2] ? Tn.shape()[ax1] : Tn.shape()[ax2];

      Tensor out = Tensor({Nelem}, Tn.dtype(), Tn.device());
      out.storage().set_zeros();

#ifdef UNI_OMP
      int Nomp = 1;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0) Nomp = omp_get_num_threads();
      }
      // std::cout << Nomp <<std::endl;

      if (shape.size() == 0) {
        switch (Tn.dtype()) {
          case Type.ComplexDouble:
            _trace_2d_para<cytnx_complex128>(out, Tn, Ndiag, Nomp);
            break;
          case Type.ComplexFloat:
            _trace_2d_para<cytnx_complex64>(out, Tn, Ndiag, Nomp);
            break;
          case Type.Double:
            _trace_2d_para<cytnx_double>(out, Tn, Ndiag, Nomp);
            break;
          case Type.Float:
            _trace_2d_para<cytnx_float>(out, Tn, Ndiag, Nomp);
            break;
          case Type.Uint64:
            _trace_2d_para<cytnx_uint64>(out, Tn, Ndiag, Nomp);
            break;
          case Type.Int64:
            _trace_2d_para<cytnx_int64>(out, Tn, Ndiag, Nomp);
            break;
          case Type.Uint32:
            _trace_2d_para<cytnx_uint32>(out, Tn, Ndiag, Nomp);
            break;
          case Type.Int32:
            _trace_2d_para<cytnx_int32>(out, Tn, Ndiag, Nomp);
            break;
          case Type.Int16:
            _trace_2d_para<cytnx_int16>(out, Tn, Ndiag, Nomp);
            break;
          case Type.Uint16:
            _trace_2d_para<cytnx_uint16>(out, Tn, Ndiag, Nomp);
            break;
          case Type.Bool:
            //_trace_2d_para<cytnx_bool>(out,Tn,Ndiag,Nomp);
            cytnx_error_msg(
              true,
              "[ERROR][Trace] Bool type cannot perform Trace, use .astype() to promote first.%s",
              "\n");
            break;
          default:
            cytnx_error_msg(true, "[ERROR][Trace] invalid Type.%s", "\n");
            break;
        }
      } else {
        vector<cytnx_uint64> remain_rank_id;
        vector<cytnx_uint64> accu(shape.size());
        accu.back() = 1;
        for (int i = shape.size() - 1; i > 0; i--) accu[i - 1] = accu[i] * shape[i];

        for (cytnx_uint64 i = 0; i < Tn.shape().size(); i++) {
          if (i != ax1 && i != ax2) remain_rank_id.push_back(i);
        }

        switch (Tn.dtype()) {
          case Type.ComplexDouble:
            _trace_nd_para<cytnx_complex128>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape,
                                             ax1, ax2, Nomp);
            break;
          case Type.ComplexFloat:
            _trace_nd_para<cytnx_complex64>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1,
                                            ax2, Nomp);
            break;
          case Type.Double:
            _trace_nd_para<cytnx_double>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1,
                                         ax2, Nomp);
            break;
          case Type.Float:
            _trace_nd_para<cytnx_float>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1,
                                        ax2, Nomp);
            break;
          case Type.Uint64:
            _trace_nd_para<cytnx_uint64>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1,
                                         ax2, Nomp);
            break;
          case Type.Int64:
            _trace_nd_para<cytnx_int64>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1,
                                        ax2, Nomp);
            break;
          case Type.Int32:
            _trace_nd_para<cytnx_int32>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1,
                                        ax2, Nomp);
            break;
          case Type.Uint32:
            _trace_nd_para<cytnx_uint32>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1,
                                         ax2, Nomp);
            break;
          case Type.Uint16:
            _trace_nd_para<cytnx_uint16>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1,
                                         ax2, Nomp);
            break;
          case Type.Int16:
            _trace_nd_para<cytnx_int16>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1,
                                        ax2, Nomp);
            break;
          case Type.Bool:
            cytnx_error_msg(
              true,
              "[ERROR][Trace] Bool type cannot perform Trace, use .astype() to promote first.%s",
              "\n");
            break;
          default:
            cytnx_error_msg(true, "[ERROR][Trace] Invalid Type.%s", "\n");
            break;
        }  // switch
        out.reshape_(shape);
      }

#else

      if (shape.size() == 0) {
        switch (Tn.dtype()) {
          case Type.ComplexDouble:
            _trace_2d<cytnx_complex128>(out, Tn, Ndiag);
            break;
          case Type.ComplexFloat:
            _trace_2d<cytnx_complex64>(out, Tn, Ndiag);
            break;
          case Type.Double:
            _trace_2d<cytnx_double>(out, Tn, Ndiag);
            break;
          case Type.Float:
            _trace_2d<cytnx_float>(out, Tn, Ndiag);
            break;
          case Type.Uint64:
            _trace_2d<cytnx_uint64>(out, Tn, Ndiag);
            break;
          case Type.Int64:
            _trace_2d<cytnx_int64>(out, Tn, Ndiag);
            break;
          case Type.Uint32:
            _trace_2d<cytnx_uint32>(out, Tn, Ndiag);
            break;
          case Type.Int32:
            _trace_2d<cytnx_int32>(out, Tn, Ndiag);
            break;
          case Type.Int16:
            _trace_2d<cytnx_int16>(out, Tn, Ndiag);
            break;
          case Type.Uint16:
            _trace_2d<cytnx_uint16>(out, Tn, Ndiag);
            break;
          case Type.Bool:
            cytnx_error_msg(
              true,
              "[ERROR][Trace] Bool type cannot perform Trace, use .astype() to promote first.%s",
              "\n");
            break;
          default:
            cytnx_error_msg(true, "[ERROR][Trace] invalid Type.%s", "\n");
            break;
        }
      } else {
        vector<cytnx_uint64> remain_rank_id;
        vector<cytnx_uint64> accu(shape.size());
        accu.back() = 1;
        for (int i = shape.size() - 1; i > 0; i--) accu[i - 1] = accu[i] * shape[i];

        for (cytnx_uint64 i = 0; i < Tn.shape().size(); i++) {
          if (i != ax1 && i != ax2) remain_rank_id.push_back(i);
        }

        switch (Tn.dtype()) {
          case Type.ComplexDouble:
            _trace_nd<cytnx_complex128>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1,
                                        ax2);
            break;
          case Type.ComplexFloat:
            _trace_nd<cytnx_complex64>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1,
                                       ax2);
            break;
          case Type.Double:
            _trace_nd<cytnx_double>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
            break;
          case Type.Float:
            _trace_nd<cytnx_float>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
            break;
          case Type.Uint64:
            _trace_nd<cytnx_uint64>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
            break;
          case Type.Int64:
            _trace_nd<cytnx_int64>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
            break;
          case Type.Int32:
            _trace_nd<cytnx_int32>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
            break;
          case Type.Uint32:
            _trace_nd<cytnx_uint32>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
            break;
          case Type.Uint16:
            _trace_nd<cytnx_uint16>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
            break;
          case Type.Int16:
            _trace_nd<cytnx_int16>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
            break;
          case Type.Bool:
            cytnx_error_msg(
              true,
              "[ERROR][Trace] Bool type cannot perform Trace, use .astype() to promote first.%s",
              "\n");
            break;
          default:
            cytnx_error_msg(true, "[ERROR][Trace] Invalid Type.%s", "\n");
            break;
        }  // switch
        out.reshape_(shape);
      }

#endif

      return out;
    }

  }  // namespace linalg
}  // namespace cytnx
