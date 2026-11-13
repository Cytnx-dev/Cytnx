#include "cuSvd_internal.hpp"
#include "backend/linalg_internal_interface.hpp"
#include "backend/utils_internal_gpu/cuScopedResource_gpu.hpp"

#include <cstddef>
#include <vector>
#include "backend/utils_internal_gpu/cuLibraryHandle_gpu.hpp"

namespace cytnx {

  namespace linalg_internal {

    // Orientation and workspace conventions shared by the four cuSvd_internal_* variants.
    //
    // cusolverDnXgesvdp takes a column-major matrix. Cytnx's row-major M x N buffer, read
    // column-major, is the N x M transpose A^T. For M <= N that buffer is passed as it is: with
    // A^T = U' S V'^H, Cytnx's vT is U'^T (U' read row-major) and Cytnx's U is conj(V'), obtained
    // by transposing the V' buffer to row-major and conjugating it.
    //
    // For M > N the same call would hand cuSOLVER a wide matrix, for which gesvdp requests a
    // workspace of about 4 * M^2 elements whatever N is and fails outright for large M (from
    // roughly 33000 with CUDA 12.9). The scratch copy is therefore transposed first, so that
    // cuSOLVER sees the tall M x N matrix A itself, whose workspace is linear in M for fixed N.
    // With A = U' S V'^H, Cytnx's U is U' transposed to row-major and Cytnx's vT is conj(V' read
    // row-major).
    //
    // The device workspace size returned by cusolverDnXgesvdp_bufferSize is a byte count
    // ("workspaceInBytesOnDevice"), not an element count as in the legacy cusolverDn<t> API.

    /// cuSvd
    void cuSvd_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &U,
                           boost::intrusive_ptr<Storage_base> &vT,
                           boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                           const cytnx_int64 &N) {
      using data_type = cytnx_complex128;
      cudaDataType cuda_data_type = CUDA_C_64F;
      cudaDataType cuda_data_typeR = CUDA_R_64F;
      assert(sizeof(cuDoubleComplex) == sizeof(cytnx_complex128));

      cusolverEigMode_t jobz;
      // if U and vT are void, then it will not be computed.
      jobz = (U->dtype() == Type.Void and vT->dtype() == Type.Void) ? CUSOLVER_EIG_MODE_NOVECTOR
                                                                    : CUSOLVER_EIG_MODE_VECTOR;

      // const int econ = 0; /* i.e. 'A' in gesvd  */
      cytnx_int32 econ = 1; /* i.e. 'S' in gesvd  */

      // create handles:
      // Scoped resources (#1146): the info check below throws, so ownership -- not the cleanup
      // block that used to sit at the tail of this function -- is what prevents a leak.
      // Shared per-device handle (#1144): cusolverDnCreate costs ~456 us per call
      // and its ~12 MB device workspace was reallocated every time.
      cusolverDnHandle_t cusolverH = utils_internal::get_cusolverdn_handle();

      // Scratch copy of the input in the orientation given to cuSOLVER (see the file comment).
      const bool is_tall = M > N;
      boost::intrusive_ptr<Storage_base> tall_in;
      utils_internal::DeviceBuffer<data_type> wide_in;
      void *solver_in = nullptr;
      if (is_tall) {
        tall_in = in->Move_memory({(cytnx_uint64)M, (cytnx_uint64)N}, {1, 0}, {1, 0});
        solver_in = tall_in->data();
      } else {
        wide_in = utils_internal::DeviceBuffer<data_type>(M * N);
        checkCudaErrors(cudaMemcpy(wide_in.get(), in->data(), sizeof(data_type) * M * N,
                                   cudaMemcpyDeviceToDevice));
        solver_in = wide_in.get();
      }
      const cytnx_int64 solver_rows = is_tall ? M : N;
      const cytnx_int64 solver_cols = is_tall ? N : M;
      cytnx_int64 min = std::min(M, N);

      // UMem/vTMem alias U's and vT's storage when it exists and are temporaries otherwise; the
      // owned_* buffers are non-empty only in the temporary case. The temporaries are needed even
      // when no singular vectors are requested: cusolverDnXgesvdp accesses the U and V buffers
      // with CUSOLVER_EIG_MODE_NOVECTOR too, so passing nullptr is an illegal device access (#831).
      // They use the economy sizes (econ = 1), M * min and N * min, which are the sizes of the
      // caller-supplied U and vT storage in the vector case.
      void *UMem = nullptr, *vTMem = nullptr;
      utils_internal::DeviceBuffer<data_type> owned_u_mem, owned_vt_mem;
      if (U->data()) {
        UMem = U->data();
      } else {
        owned_u_mem = utils_internal::DeviceBuffer<data_type>(M * min);
        UMem = owned_u_mem.get();
      }
      if (vT->data()) {
        vTMem = vT->data();
      } else {
        owned_vt_mem = utils_internal::DeviceBuffer<data_type>(N * min);
        vTMem = owned_vt_mem.get();
      }

      // cuSOLVER's U is Cytnx's U for a tall input and Cytnx's vT otherwise.
      void *solver_u = is_tall ? UMem : vTMem;
      void *solver_v = is_tall ? vTMem : UMem;

      std::size_t d_lwork = 0; /* size of device workspace in bytes */
      void *d_work = nullptr; /* device workspace for getrf */
      std::size_t h_lwork = 0; /* size of workspace */
      void *h_work = nullptr; /* host workspace for getrf */
      cytnx_double h_err_sigma;
      // query working space :
      checkCudaErrors(cusolverDnXgesvdp_bufferSize(
        cusolverH, nullptr, /* params */
        jobz, econ, solver_rows, solver_cols, cuda_data_type, /* dataTypeA */
        solver_in, solver_rows, cuda_data_typeR, /* dataTypeS */
        S->data(), cuda_data_type, /* dataTypeU */
        solver_u, solver_rows, /* ldu */
        cuda_data_type, /* dataTypeV */
        solver_v, solver_cols, /* ldv */
        cuda_data_type, /* computeType */
        &d_lwork, &h_lwork));

      // allocate working space:
      utils_internal::DeviceBuffer<std::byte> owned_d_work(d_lwork);
      d_work = owned_d_work.get();
      std::vector<char> owned_h_work;
      if (0 < h_lwork) {
        owned_h_work.resize(h_lwork);
        h_work = owned_h_work.data();
        if (d_work == nullptr) {
          throw std::runtime_error("Error: d_work not allocated.");
        }
      }

      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      checkCudaErrors(cudaMemset(devinfo.get(), 0, sizeof(cytnx_int32)));

      cytnx_int32 info;
      /// compute:
      checkCudaErrors(
        cusolverDnXgesvdp(cusolverH, nullptr, /* params */
                          jobz, econ, solver_rows, solver_cols, cuda_data_type, /* dataTypeA */
                          solver_in, solver_rows, cuda_data_typeR, /* dataTypeS */
                          S->data(), cuda_data_type, /* dataTypeU */
                          solver_u, solver_rows, /* ldu */
                          cuda_data_type, /* dataTypeV */
                          solver_v, solver_cols, /* ldv */
                          cuda_data_type, /* computeType */
                          d_work, d_lwork, h_work, h_lwork, devinfo.get(), &h_err_sigma));
      if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        // The buffer that received cuSOLVER's V holds V^T once read row-major; Cytnx needs V^H.
        if (is_tall) {
          linalg_internal::cuConj_inplace_dispatch(vT, N * min);
        } else {
          linalg_internal::cuConj_inplace_dispatch(U, M * min);
        }
      }
      // get info
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_warning_msg(
        h_err_sigma > 1e-12,
        "Warning: Singular values approach zero, SVD may not be accurate, err_sigma = %E\n",
        h_err_sigma);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnXgesvdp': cuBlas INFO = ", info);
    }
    void cuSvd_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &U,
                           boost::intrusive_ptr<Storage_base> &vT,
                           boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                           const cytnx_int64 &N) {
      using data_type = cytnx_complex64;
      cudaDataType cuda_data_type = CUDA_C_32F;
      cudaDataType cuda_data_typeR = CUDA_R_32F;
      assert(sizeof(cuFloatComplex) == sizeof(cytnx_complex64));

      cusolverEigMode_t jobz;
      // if U and vT are void, then it will not be computed.
      jobz = (U->dtype() == Type.Void and vT->dtype() == Type.Void) ? CUSOLVER_EIG_MODE_NOVECTOR
                                                                    : CUSOLVER_EIG_MODE_VECTOR;

      // const int econ = 0; /* i.e. 'A' in gesvd  */
      cytnx_int32 econ = 1; /* i.e. 'S' in gesvd  */

      // create handles:
      // Scoped resources (#1146): the info check below throws, so ownership -- not the cleanup
      // block that used to sit at the tail of this function -- is what prevents a leak.
      // Shared per-device handle (#1144): cusolverDnCreate costs ~456 us per call
      // and its ~12 MB device workspace was reallocated every time.
      cusolverDnHandle_t cusolverH = utils_internal::get_cusolverdn_handle();

      // Scratch copy of the input in the orientation given to cuSOLVER (see the file comment).
      const bool is_tall = M > N;
      boost::intrusive_ptr<Storage_base> tall_in;
      utils_internal::DeviceBuffer<data_type> wide_in;
      void *solver_in = nullptr;
      if (is_tall) {
        tall_in = in->Move_memory({(cytnx_uint64)M, (cytnx_uint64)N}, {1, 0}, {1, 0});
        solver_in = tall_in->data();
      } else {
        wide_in = utils_internal::DeviceBuffer<data_type>(M * N);
        checkCudaErrors(cudaMemcpy(wide_in.get(), in->data(), sizeof(data_type) * M * N,
                                   cudaMemcpyDeviceToDevice));
        solver_in = wide_in.get();
      }
      const cytnx_int64 solver_rows = is_tall ? M : N;
      const cytnx_int64 solver_cols = is_tall ? N : M;
      cytnx_int64 min = std::min(M, N);

      // UMem/vTMem alias U's and vT's storage when it exists and are temporaries otherwise; the
      // owned_* buffers are non-empty only in the temporary case. The temporaries are needed even
      // when no singular vectors are requested: cusolverDnXgesvdp accesses the U and V buffers
      // with CUSOLVER_EIG_MODE_NOVECTOR too, so passing nullptr is an illegal device access (#831).
      // They use the economy sizes (econ = 1), M * min and N * min, which are the sizes of the
      // caller-supplied U and vT storage in the vector case.
      void *UMem = nullptr, *vTMem = nullptr;
      utils_internal::DeviceBuffer<data_type> owned_u_mem, owned_vt_mem;
      if (U->data()) {
        UMem = U->data();
      } else {
        owned_u_mem = utils_internal::DeviceBuffer<data_type>(M * min);
        UMem = owned_u_mem.get();
      }
      if (vT->data()) {
        vTMem = vT->data();
      } else {
        owned_vt_mem = utils_internal::DeviceBuffer<data_type>(N * min);
        vTMem = owned_vt_mem.get();
      }

      // cuSOLVER's U is Cytnx's U for a tall input and Cytnx's vT otherwise.
      void *solver_u = is_tall ? UMem : vTMem;
      void *solver_v = is_tall ? vTMem : UMem;

      std::size_t d_lwork = 0; /* size of device workspace in bytes */
      void *d_work = nullptr; /* device workspace for getrf */
      std::size_t h_lwork = 0; /* size of workspace */
      void *h_work = nullptr; /* host workspace for getrf */
      cytnx_double h_err_sigma;
      // query working space :
      checkCudaErrors(cusolverDnXgesvdp_bufferSize(
        cusolverH, nullptr, /* params */
        jobz, econ, solver_rows, solver_cols, cuda_data_type, /* dataTypeA */
        solver_in, solver_rows, cuda_data_typeR, /* dataTypeS */
        S->data(), cuda_data_type, /* dataTypeU */
        solver_u, solver_rows, /* ldu */
        cuda_data_type, /* dataTypeV */
        solver_v, solver_cols, /* ldv */
        cuda_data_type, /* computeType */
        &d_lwork, &h_lwork));

      // allocate working space:
      utils_internal::DeviceBuffer<std::byte> owned_d_work(d_lwork);
      d_work = owned_d_work.get();
      std::vector<char> owned_h_work;
      if (0 < h_lwork) {
        owned_h_work.resize(h_lwork);
        h_work = owned_h_work.data();
        if (d_work == nullptr) {
          throw std::runtime_error("Error: d_work not allocated.");
        }
      }

      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      checkCudaErrors(cudaMemset(devinfo.get(), 0, sizeof(cytnx_int32)));

      cytnx_int32 info;
      /// compute:
      checkCudaErrors(
        cusolverDnXgesvdp(cusolverH, nullptr, /* params */
                          jobz, econ, solver_rows, solver_cols, cuda_data_type, /* dataTypeA */
                          solver_in, solver_rows, cuda_data_typeR, /* dataTypeS */
                          S->data(), cuda_data_type, /* dataTypeU */
                          solver_u, solver_rows, /* ldu */
                          cuda_data_type, /* dataTypeV */
                          solver_v, solver_cols, /* ldv */
                          cuda_data_type, /* computeType */
                          d_work, d_lwork, h_work, h_lwork, devinfo.get(), &h_err_sigma));
      if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        // The buffer that received cuSOLVER's V holds V^T once read row-major; Cytnx needs V^H.
        if (is_tall) {
          linalg_internal::cuConj_inplace_dispatch(vT, N * min);
        } else {
          linalg_internal::cuConj_inplace_dispatch(U, M * min);
        }
      }
      // get info
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_warning_msg(
        h_err_sigma > 1e-12,
        "Warning: Singular values approach zero, SVD may not be accurate, err_sigma = %E\n",
        h_err_sigma);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnXgesvdp': cuBlas INFO = ", info);
    }
    void cuSvd_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &U,
                          boost::intrusive_ptr<Storage_base> &vT,
                          boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                          const cytnx_int64 &N) {
      using data_type = cytnx_double;
      cudaDataType cuda_data_type = CUDA_R_64F;
      cudaDataType cuda_data_typeR = CUDA_R_64F;

      cusolverEigMode_t jobz;
      // if U and vT are void, then it will not be computed.
      jobz = (U->dtype() == Type.Void and vT->dtype() == Type.Void) ? CUSOLVER_EIG_MODE_NOVECTOR
                                                                    : CUSOLVER_EIG_MODE_VECTOR;

      // const int econ = 0; /* i.e. 'A' in gesvd  */
      cytnx_int32 econ = 1; /* i.e. 'S' in gesvd  */

      // create handles:
      // Scoped resources (#1146): the info check below throws, so ownership -- not the cleanup
      // block that used to sit at the tail of this function -- is what prevents a leak.
      // Shared per-device handle (#1144): cusolverDnCreate costs ~456 us per call
      // and its ~12 MB device workspace was reallocated every time.
      cusolverDnHandle_t cusolverH = utils_internal::get_cusolverdn_handle();

      // Scratch copy of the input in the orientation given to cuSOLVER (see the file comment).
      const bool is_tall = M > N;
      boost::intrusive_ptr<Storage_base> tall_in;
      utils_internal::DeviceBuffer<data_type> wide_in;
      void *solver_in = nullptr;
      if (is_tall) {
        tall_in = in->Move_memory({(cytnx_uint64)M, (cytnx_uint64)N}, {1, 0}, {1, 0});
        solver_in = tall_in->data();
      } else {
        wide_in = utils_internal::DeviceBuffer<data_type>(M * N);
        checkCudaErrors(cudaMemcpy(wide_in.get(), in->data(), sizeof(data_type) * M * N,
                                   cudaMemcpyDeviceToDevice));
        solver_in = wide_in.get();
      }
      const cytnx_int64 solver_rows = is_tall ? M : N;
      const cytnx_int64 solver_cols = is_tall ? N : M;
      cytnx_int64 min = std::min(M, N);

      // UMem/vTMem alias U's and vT's storage when it exists and are temporaries otherwise; the
      // owned_* buffers are non-empty only in the temporary case. The temporaries are needed even
      // when no singular vectors are requested: cusolverDnXgesvdp accesses the U and V buffers
      // with CUSOLVER_EIG_MODE_NOVECTOR too, so passing nullptr is an illegal device access (#831).
      // They use the economy sizes (econ = 1), M * min and N * min, which are the sizes of the
      // caller-supplied U and vT storage in the vector case.
      void *UMem = nullptr, *vTMem = nullptr;
      utils_internal::DeviceBuffer<data_type> owned_u_mem, owned_vt_mem;
      if (U->data()) {
        UMem = U->data();
      } else {
        owned_u_mem = utils_internal::DeviceBuffer<data_type>(M * min);
        UMem = owned_u_mem.get();
      }
      if (vT->data()) {
        vTMem = vT->data();
      } else {
        owned_vt_mem = utils_internal::DeviceBuffer<data_type>(N * min);
        vTMem = owned_vt_mem.get();
      }

      // cuSOLVER's U is Cytnx's U for a tall input and Cytnx's vT otherwise.
      void *solver_u = is_tall ? UMem : vTMem;
      void *solver_v = is_tall ? vTMem : UMem;

      std::size_t d_lwork = 0; /* size of device workspace in bytes */
      void *d_work = nullptr; /* device workspace for getrf */
      std::size_t h_lwork = 0; /* size of workspace */
      void *h_work = nullptr; /* host workspace for getrf */
      cytnx_double h_err_sigma;
      // query working space :
      checkCudaErrors(cusolverDnXgesvdp_bufferSize(
        cusolverH, nullptr, /* params */
        jobz, econ, solver_rows, solver_cols, cuda_data_type, /* dataTypeA */
        solver_in, solver_rows, cuda_data_typeR, /* dataTypeS */
        S->data(), cuda_data_type, /* dataTypeU */
        solver_u, solver_rows, /* ldu */
        cuda_data_type, /* dataTypeV */
        solver_v, solver_cols, /* ldv */
        cuda_data_type, /* computeType */
        &d_lwork, &h_lwork));

      // allocate working space:
      utils_internal::DeviceBuffer<std::byte> owned_d_work(d_lwork);
      d_work = owned_d_work.get();
      std::vector<char> owned_h_work;
      if (0 < h_lwork) {
        owned_h_work.resize(h_lwork);
        h_work = owned_h_work.data();
        if (d_work == nullptr) {
          throw std::runtime_error("Error: d_work not allocated.");
        }
      }

      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      checkCudaErrors(cudaMemset(devinfo.get(), 0, sizeof(cytnx_int32)));

      cytnx_int32 info;
      /// compute:
      checkCudaErrors(
        cusolverDnXgesvdp(cusolverH, nullptr, /* params */
                          jobz, econ, solver_rows, solver_cols, cuda_data_type, /* dataTypeA */
                          solver_in, solver_rows, cuda_data_typeR, /* dataTypeS */
                          S->data(), cuda_data_type, /* dataTypeU */
                          solver_u, solver_rows, /* ldu */
                          cuda_data_type, /* dataTypeV */
                          solver_v, solver_cols, /* ldv */
                          cuda_data_type, /* computeType */
                          d_work, d_lwork, h_work, h_lwork, devinfo.get(), &h_err_sigma));
      if (jobz == CUSOLVER_EIG_MODE_VECTOR)
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});

      // get info
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_warning_msg(
        h_err_sigma > 1e-12,
        "Warning: Singular values approach zero, SVD may not be accurate, err_sigma = %E\n",
        h_err_sigma);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnXgesvdp': cuBlas INFO = ", info);
    }
    void cuSvd_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &U,
                          boost::intrusive_ptr<Storage_base> &vT,
                          boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                          const cytnx_int64 &N) {
      using data_type = cytnx_float;
      cudaDataType cuda_data_type = CUDA_R_32F;
      cudaDataType cuda_data_typeR = CUDA_R_32F;

      cusolverEigMode_t jobz;
      // if U and vT are void, then it will not be computed.
      jobz = (U->dtype() == Type.Void and vT->dtype() == Type.Void) ? CUSOLVER_EIG_MODE_NOVECTOR
                                                                    : CUSOLVER_EIG_MODE_VECTOR;

      // const int econ = 0; /* i.e. 'A' in gesvd  */
      cytnx_int32 econ = 1; /* i.e. 'S' in gesvd  */

      // create handles:
      // Scoped resources (#1146): the info check below throws, so ownership -- not the cleanup
      // block that used to sit at the tail of this function -- is what prevents a leak.
      // Shared per-device handle (#1144): cusolverDnCreate costs ~456 us per call
      // and its ~12 MB device workspace was reallocated every time.
      cusolverDnHandle_t cusolverH = utils_internal::get_cusolverdn_handle();

      // Scratch copy of the input in the orientation given to cuSOLVER (see the file comment).
      const bool is_tall = M > N;
      boost::intrusive_ptr<Storage_base> tall_in;
      utils_internal::DeviceBuffer<data_type> wide_in;
      void *solver_in = nullptr;
      if (is_tall) {
        tall_in = in->Move_memory({(cytnx_uint64)M, (cytnx_uint64)N}, {1, 0}, {1, 0});
        solver_in = tall_in->data();
      } else {
        wide_in = utils_internal::DeviceBuffer<data_type>(M * N);
        checkCudaErrors(cudaMemcpy(wide_in.get(), in->data(), sizeof(data_type) * M * N,
                                   cudaMemcpyDeviceToDevice));
        solver_in = wide_in.get();
      }
      const cytnx_int64 solver_rows = is_tall ? M : N;
      const cytnx_int64 solver_cols = is_tall ? N : M;
      cytnx_int64 min = std::min(M, N);

      // UMem/vTMem alias U's and vT's storage when it exists and are temporaries otherwise; the
      // owned_* buffers are non-empty only in the temporary case. The temporaries are needed even
      // when no singular vectors are requested: cusolverDnXgesvdp accesses the U and V buffers
      // with CUSOLVER_EIG_MODE_NOVECTOR too, so passing nullptr is an illegal device access (#831).
      // They use the economy sizes (econ = 1), M * min and N * min, which are the sizes of the
      // caller-supplied U and vT storage in the vector case.
      void *UMem = nullptr, *vTMem = nullptr;
      utils_internal::DeviceBuffer<data_type> owned_u_mem, owned_vt_mem;
      if (U->data()) {
        UMem = U->data();
      } else {
        owned_u_mem = utils_internal::DeviceBuffer<data_type>(M * min);
        UMem = owned_u_mem.get();
      }
      if (vT->data()) {
        vTMem = vT->data();
      } else {
        owned_vt_mem = utils_internal::DeviceBuffer<data_type>(N * min);
        vTMem = owned_vt_mem.get();
      }

      // cuSOLVER's U is Cytnx's U for a tall input and Cytnx's vT otherwise.
      void *solver_u = is_tall ? UMem : vTMem;
      void *solver_v = is_tall ? vTMem : UMem;

      std::size_t d_lwork = 0; /* size of device workspace in bytes */
      void *d_work = nullptr; /* device workspace for getrf */
      std::size_t h_lwork = 0; /* size of workspace */
      void *h_work = nullptr; /* host workspace for getrf */
      cytnx_double h_err_sigma;
      // query working space :
      checkCudaErrors(cusolverDnXgesvdp_bufferSize(
        cusolverH, nullptr, /* params */
        jobz, econ, solver_rows, solver_cols, cuda_data_type, /* dataTypeA */
        solver_in, solver_rows, cuda_data_typeR, /* dataTypeS */
        S->data(), cuda_data_type, /* dataTypeU */
        solver_u, solver_rows, /* ldu */
        cuda_data_type, /* dataTypeV */
        solver_v, solver_cols, /* ldv */
        cuda_data_type, /* computeType */
        &d_lwork, &h_lwork));

      // allocate working space:
      utils_internal::DeviceBuffer<std::byte> owned_d_work(d_lwork);
      d_work = owned_d_work.get();
      std::vector<char> owned_h_work;
      if (0 < h_lwork) {
        owned_h_work.resize(h_lwork);
        h_work = owned_h_work.data();
        if (d_work == nullptr) {
          throw std::runtime_error("Error: d_work not allocated.");
        }
      }

      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      checkCudaErrors(cudaMemset(devinfo.get(), 0, sizeof(cytnx_int32)));

      cytnx_int32 info;
      /// compute:
      checkCudaErrors(
        cusolverDnXgesvdp(cusolverH, nullptr, /* params */
                          jobz, econ, solver_rows, solver_cols, cuda_data_type, /* dataTypeA */
                          solver_in, solver_rows, cuda_data_typeR, /* dataTypeS */
                          S->data(), cuda_data_type, /* dataTypeU */
                          solver_u, solver_rows, /* ldu */
                          cuda_data_type, /* dataTypeV */
                          solver_v, solver_cols, /* ldv */
                          cuda_data_type, /* computeType */
                          d_work, d_lwork, h_work, h_lwork, devinfo.get(), &h_err_sigma));
      if (jobz == CUSOLVER_EIG_MODE_VECTOR)
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});

      // get info
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_warning_msg(
        h_err_sigma > 1e-12,
        "Warning: Singular values approach zero, SVD may not be accurate, err_sigma = %E\n",
        h_err_sigma);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnXgesvdp': cuBlas INFO = ", info);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
