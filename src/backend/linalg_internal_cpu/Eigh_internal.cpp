#include "Eigh_internal.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>

#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"

namespace cytnx {

  namespace linalg_internal {

    namespace {

      // One overload per dtype: runs the requested LAPACK driver on the column-major n x n
      // matrix `a` (upper triangle), writing the eigenvalues to `w` and, for jobz == 'V', the
      // eigenvectors over `a`.
      lapack_int run_driver(EighDriver driver, char jobz, lapack_int n, double *a, double *w) {
        if (driver == EighDriver::kDc)
          return LAPACKE_dsyevd(LAPACK_COL_MAJOR, jobz, 'U', n, a, n, w);
        return LAPACKE_dsyev(LAPACK_COL_MAJOR, jobz, 'U', n, a, n, w);
      }
      lapack_int run_driver(EighDriver driver, char jobz, lapack_int n, float *a, float *w) {
        if (driver == EighDriver::kDc)
          return LAPACKE_ssyevd(LAPACK_COL_MAJOR, jobz, 'U', n, a, n, w);
        return LAPACKE_ssyev(LAPACK_COL_MAJOR, jobz, 'U', n, a, n, w);
      }
      lapack_int run_driver(EighDriver driver, char jobz, lapack_int n, std::complex<double> *a,
                            double *w) {
        auto *la = reinterpret_cast<lapack_complex_double *>(a);
        if (driver == EighDriver::kDc)
          return LAPACKE_zheevd(LAPACK_COL_MAJOR, jobz, 'U', n, la, n, w);
        return LAPACKE_zheev(LAPACK_COL_MAJOR, jobz, 'U', n, la, n, w);
      }
      lapack_int run_driver(EighDriver driver, char jobz, lapack_int n, std::complex<float> *a,
                            float *w) {
        auto *la = reinterpret_cast<lapack_complex_float *>(a);
        if (driver == EighDriver::kDc)
          return LAPACKE_cheevd(LAPACK_COL_MAJOR, jobz, 'U', n, la, n, w);
        return LAPACKE_cheev(LAPACK_COL_MAJOR, jobz, 'U', n, la, n, w);
      }

      // T is the element type of the matrix, Real the type of its eigenvalues.
      template <class T, class Real>
      void eigh_internal(const boost::intrusive_ptr<Storage_base> &in,
                         boost::intrusive_ptr<Storage_base> &e,
                         boost::intrusive_ptr<Storage_base> &v, cytnx_int64 L, EighDriver driver) {
        const std::size_t count = static_cast<std::size_t>(L) * static_cast<std::size_t>(L);
        const T *input = static_cast<const T *>(in->data());
        const bool want_vectors = v->dtype() != Type.Void;

        // LAPACK overwrites the matrix, with the eigenvectors when they are requested: work in
        // v's storage then, and in a scratch copy otherwise.
        std::vector<T> scratch;
        T *a;
        if (want_vectors) {
          a = static_cast<T *>(v->data());
          std::copy(input, input + count, a);
        } else {
          scratch.assign(input, input + count);
          a = scratch.data();
        }
        const char jobz = want_vectors ? 'V' : 'N';
        Real *w = static_cast<Real *>(e->data());

        lapack_int info = run_driver(driver, jobz, L, a, w);
        if (driver == EighDriver::kDc && info > 0) {
          // LAPACK's own advice for a divide-and-conquer failure: redo with QR iteration.
          cytnx_warning_msg(true,
                            "[Eigh] the divide-and-conquer driver failed (Lapack INFO = %lld); "
                            "falling back to the QR driver.\n",
                            static_cast<long long>(info));
          std::copy(input, input + count, a);
          driver = EighDriver::kQr;
          info = run_driver(driver, jobz, L, a, w);
        }
        cytnx_error_msg(
          info != 0, "Error in the Lapack Hermitian eigensolver (%s driver): Lapack INFO = %lld\n",
          driver == EighDriver::kDc ? "divide-and-conquer" : "QR", static_cast<long long>(info));
      }

    }  // namespace

    /// Eigh
    void Eigh_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &e,
                          boost::intrusive_ptr<Storage_base> &v, cytnx_int64 L, EighDriver driver) {
      eigh_internal<std::complex<double>, double>(in, e, v, L, driver);
    }
    void Eigh_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &e,
                          boost::intrusive_ptr<Storage_base> &v, cytnx_int64 L, EighDriver driver) {
      eigh_internal<std::complex<float>, float>(in, e, v, L, driver);
    }
    void Eigh_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                         boost::intrusive_ptr<Storage_base> &e,
                         boost::intrusive_ptr<Storage_base> &v, cytnx_int64 L, EighDriver driver) {
      eigh_internal<double, double>(in, e, v, L, driver);
    }
    void Eigh_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                         boost::intrusive_ptr<Storage_base> &e,
                         boost::intrusive_ptr<Storage_base> &v, cytnx_int64 L, EighDriver driver) {
      eigh_internal<float, float>(in, e, v, L, driver);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
