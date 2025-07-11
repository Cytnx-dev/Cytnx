#include "linalg.hpp"
#include "Generator.hpp"
#include "random.hpp"
#include "Tensor.hpp"
#include "LinOp.hpp"
#include <cfloat>
#include <functional>
#include <algorithm>

#include "backend/arpack_wrapper.hpp"

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  namespace linalg {
    typedef Accessor ac;

    // T_ten: Tensor or UniTensor
    template <typename T, typename T_ten>
    static T* get_obj_data_ptr(const T_ten& buffer, const cytnx_int32 bk_idx = 0) {
      if constexpr (std::is_same_v<T_ten, UniTensor>) {
        if (buffer.uten_type() == UTenType.Block) {
          if (buffer.device() == Device.cpu) {
            return buffer.get_blocks_()[bk_idx].template ptr_as<T>();
          } else {  // on cuda
  #ifdef UNI_GPU
            return reinterpret_cast<T*>(
              buffer.get_blocks_()[bk_idx].template gpu_ptr_as<void>(false));
  #endif
          }
        } else if (buffer.uten_type() == UTenType.Dense) {
          if (buffer.device() == Device.cpu) {
            return buffer.get_block_().template ptr_as<T>();
          } else {  // on cuda
  #ifdef UNI_GPU
            return reinterpret_cast<T*>(buffer.get_block_().template gpu_ptr_as<void>(false));
  #endif
          }
        }
      } else if constexpr (std::is_same_v<T_ten, Tensor>) {
        if (buffer.device() == Device.cpu) {
          return buffer.template ptr_as<T>();
        } else {  // cuda
  #ifdef UNI_GPU
          return reinterpret_cast<T*>(buffer.template gpu_ptr_as<void>(false));
  #endif
        }
      }
    }

    static cytnx_int64 get_dim(const Tensor& T) {
      auto shape = T.shape();
      cytnx_int64 dim = 1;
      for (auto& x : shape) {
        dim *= x;
      }
      return dim;
    }

    static cytnx_int64 get_elem_num(const UniTensor& UT) {
      if (UT.uten_type() == UTenType.Block) {
        cytnx_int64 dim = 0;
        auto& blocks = UT.get_blocks_();
        for (int i = 0; i < blocks.size(); ++i) {
          dim += get_dim(blocks[i]);
        }
        return dim;
      } else if (UT.uten_type() == UTenType.Dense) {
        return get_dim(UT.get_block_());
      }
    }

    template <typename T, typename T_ten>
    static void pass_data_UT(T_ten& UT, T* data_ptr, bool to_UT) {
      auto device = UT.device();
      if constexpr (std::is_same_v<T_ten, UniTensor>) {
        if (UT.uten_type() == UTenType.Block) {
          auto& blocks = UT.get_blocks_();
          for (auto& block : blocks) {
            auto dim = get_dim(block);
            T* UT_data = get_obj_data_ptr<T, Tensor>(block);
            if (to_UT) {
              if (device == Device.cpu) {
                memcpy(UT_data, data_ptr, dim * sizeof(T));
              } else {
  #ifdef UNI_GPU
                checkCudaErrors(
                  cudaMemcpy(UT_data, data_ptr, dim * sizeof(T), cudaMemcpyHostToDevice));
  #endif
              }
            } else {
              if (device == Device.cpu) {
                memcpy(data_ptr, UT_data, dim * sizeof(T));
              } else {
  #ifdef UNI_GPU
                checkCudaErrors(
                  cudaMemcpy(data_ptr, UT_data, dim * sizeof(T), cudaMemcpyDeviceToHost));
  #endif
              }
            }
            data_ptr += dim;
          }
          return;
        } else if (UT.uten_type() == UTenType.Dense) {
          auto& block = UT.get_block_();
          auto dim = get_dim(block);
          T* UT_data = get_obj_data_ptr<T, Tensor>(block);
          if (to_UT) {
            if (device == Device.cpu) {
              memcpy(UT_data, data_ptr, dim * sizeof(T));
            } else {
  #ifdef UNI_GPU
              checkCudaErrors(
                cudaMemcpy(UT_data, data_ptr, dim * sizeof(T), cudaMemcpyHostToDevice));
  #endif
            }
          } else {
            if (device == Device.cpu) {
              memcpy(data_ptr, UT_data, dim * sizeof(T));
            } else {
  #ifdef UNI_GPU
              checkCudaErrors(
                cudaMemcpy(data_ptr, UT_data, dim * sizeof(T), cudaMemcpyDeviceToHost));
  #endif
            }
          }
          return;
        }
      } else if constexpr (std::is_same_v<T_ten, Tensor>) {
        auto dim = get_dim(UT);
        T* UT_data = get_obj_data_ptr<T, Tensor>(UT);
        if (to_UT) {
          if (device == Device.cpu) {
            memcpy(UT_data, data_ptr, dim * sizeof(T));
          } else {
  #ifdef UNI_GPU
            checkCudaErrors(cudaMemcpy(UT_data, data_ptr, dim * sizeof(T), cudaMemcpyHostToDevice));
  #endif
          }
        } else {
          if (device == Device.cpu) {
            memcpy(data_ptr, UT_data, dim * sizeof(T));
          } else {
  #ifdef UNI_GPU
            checkCudaErrors(cudaMemcpy(data_ptr, UT_data, dim * sizeof(T), cudaMemcpyDeviceToHost));
  #endif
          }
        }
        return;
      }
    }

    // T_ten: Tensor or UniTensor
    template <typename T, typename T_ten>
    static void matvec(LinOp* Hop, T_ten& buffer, T* v_in, T* v_out) {
      buffer.contiguous_();
      pass_data_UT<T, T_ten>(buffer, v_in, true);
      auto nextTens = Hop->matvec(buffer);
      cytnx_error_msg(nextTens.dtype() != Hop->dtype(),
                      "[ERROR][Arnoldi], the output dtype in the matvec is not "
                      "consistent with the one in LinOp.%s",
                      "\n");
      nextTens.contiguous_();
      pass_data_UT<T, T_ten>(nextTens, v_out, false);
    }

    template <typename T, typename T2>
    static void clean_arpack_complex_buffer(T*& resid, T*& v, T*& workd, T*& workl, T2*& rwork,
                                            cytnx_int32*& select, T*& d, T*& z, T*& workev) {
      delete[] v;
      v = nullptr;
      delete[] resid;
      resid = nullptr;
      delete[] workd;
      workd = nullptr;
      delete[] workl;
      workl = nullptr;
      delete[] rwork;
      rwork = nullptr;
      delete[] select;
      select = nullptr;
      delete[] d;
      d = nullptr;
      delete[] z;
      z = nullptr;
      delete[] workev;
      workev = nullptr;
    }

    // sort eignvalues, it seems that arpack will not sort it...
    template <typename T, typename T2>
    std::vector<cytnx_int32> sort_metric_indices(const T* d, const std::string& which,
                                                 const cytnx_int32 k) {
      char large_or_small = which[0];  //'S' or 'L'
      char metric_type = which[1];  //'M', 'R' or 'I'

      // get the metric distance
      auto metric = std::vector<T2>(k, 0);
      if (metric_type == 'M') {
        for (cytnx_int32 i = 0; i < k; ++i) metric[i] = abs(d[i]);
      } else if (metric_type == 'R') {
        for (cytnx_int32 i = 0; i < k; ++i) metric[i] = d[i].real();
      } else if (metric_type == 'I') {
        for (cytnx_int32 i = 0; i < k; ++i) metric[i] = d[i].imag();
      } else {
      }  // never

      std::vector<cytnx_int32> indices(k);
      std::iota(indices.begin(), indices.end(), 0);  // fills with 0, 1, ..., k-1

      // smallest or largest
      bool is_small = (large_or_small == 'S');
      std::sort(indices.begin(), indices.end(), [&](cytnx_int32 i, cytnx_int32 j) {
        if (is_small) {
          if (abs(metric[i] - metric[j]) > 0) return metric[i] < metric[j];
          return d[i].imag() > d[j].imag();
        } else {
          if (abs(metric[i] - metric[j]) > 0) return metric[i] > metric[j];
          return d[i].imag() > d[j].imag();
        }
      });
      return indices;
    }

    // T_ten: Tensor or UniTensor
    // T: cytnx_complex128 or cytnx_complex64
    // T2: cytnx_double or cytnx_float. If T is cytnx_complex128, it should be
    //     cytnx_double, otherwide, should be cytnx_float
    template <typename T, typename T2, typename T_ten>
    void _Arnoldi_internal_complex(std::vector<T_ten>& out, LinOp* Hop, const T_ten& UT_init,
                                   const std::string which_str, const cytnx_uint64& maxiter,
                                   const cytnx_double& CvgCrit, const cytnx_uint64& k,
                                   const cytnx_bool& is_V, const cytnx_int32& ncv_in,
                                   const cytnx_bool& verbose) {
      std::function<void(cytnx_int32 * ido, char* bmat, cytnx_int32* n, char* which,
                         cytnx_int32* nev, T2* tol, T* resid, cytnx_int32* ncv, T* v,
                         cytnx_int32* ldv, cytnx_int32* iparam, cytnx_int32* ipntr, T* workd,
                         T* workl, cytnx_int32* lworkl, T2* rwork, cytnx_int32* info)>
        func_xnaupd;
      std::function<void(cytnx_int32 * rvec, char* howmny, cytnx_int32* select, T* d, T* z,
                         cytnx_int32* ldz, T* sigma, T* workev, char* bmat, cytnx_int32* n,
                         char* which, cytnx_int32* nev, T2* tol, T* resid, cytnx_int32* ncv, T* v,
                         cytnx_int32* ldv, cytnx_int32* iparam, cytnx_int32* ipntr, T* workd,
                         T* workl, cytnx_int32* lworkl, T2* rwork, cytnx_int32* info)>
        func_xneupd;
      auto dtype = Hop->dtype();
      if constexpr (std::is_same_v<T, cytnx_complex128>) {
        func_xnaupd = arpack::znaupd_;
        func_xneupd = arpack::zneupd_;
      } else if constexpr (std::is_same_v<T, cytnx_complex64>) {
        func_xnaupd = arpack::cnaupd_;
        func_xneupd = arpack::cneupd_;
      } else {
        static_assert(std::is_same_v<T, cytnx_complex128> && std::is_same_v<T2, cytnx_double> ||
                        std::is_same_v<T, cytnx_complex64> && std::is_same_v<T2, cytnx_float>,
                      "Unsupported template types for _Arnoldi_internal_complex. T must be "
                      "std::complex<double/float> and T2 its corresponding real type.");
      }
      cytnx_int32 dim = Hop->nx();
      cytnx_int32 nev = k;
      cytnx_int32 ido = 0;  /// reverse communication parameter, must be zero before iteration
      char bmat = 'I';  ///'I': standard eigenproblem, 'G': generalized eigenproblem
      char which[2];
      std::memcpy(which, which_str.data(), 2);  /// type of asked eigenvalues
      cytnx_int32 info = 0;
      T2 tol = static_cast<T2>(CvgCrit);
      cytnx_int32 iparam[11];
      iparam[0] = 1;  /// Specifies the shift strategy (1->exact)
      iparam[2] = maxiter;  /// Maximum number of iterations
      iparam[3] = 1;  // current code only for 1
      iparam[6] = 1;  /// Sets the mode of z(c)naupd.
      cytnx_int32 ipntr[14];
      T* resid = new T[dim];
      T_ten buffer_UT = UT_init.clone();
      cytnx_bool ifinit = true;  // not allow for false, currently
      if (ifinit) {
        info = 1;
        pass_data_UT<T, T_ten>(buffer_UT, resid, false);
      } else {
        info = 0;
      }
      // If info = 0, a randomly initial residual vector is used.
      // If info = 1, resid contains the initial guess vector provided by user or from previous run
      // on output, resid contains the final residual vector

      // the number of Ritz vector, nev+2 <= ncv <= dim
      cytnx_int32 ncv = ncv_in == 0 ? std::min(dim, 2 * nev + 10) : ncv_in;
      cytnx_int32 ldv = dim;  /// leading dimension of v
      T* v = new T[dim * ncv];
      T* workd = new T[3 * dim];
      cytnx_int32 lworkl = 3 * ncv * ncv + 5 * ncv;  /// LWORKL must be at least 3*NCV**2 + 5*NCV .
      T* workl = new T[lworkl];
      T2* rwork = new T2[ncv];
      cytnx_int32* select = new cytnx_int32[ncv];
      T* d = new T[nev + 1];
      T* z = new T[dim * nev];
      T* workev = new T[2 * ncv];

      T sigma;

      /// start iteration
      while (true) {
        func_xnaupd(&ido, &bmat, &dim, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr,
                    workd, workl, &lworkl, rwork, &info);
        if (ido == -1 || ido == 1) {
          matvec(Hop, buffer_UT, &workd[ipntr[0] - 1], &workd[ipntr[1] - 1]);
        } else if (ido == 99) {
          break;
        } else {
          clean_arpack_complex_buffer(resid, v, workd, workl, rwork, select, d, z, workev);
          cytnx_error_msg(true, "[ERROR][Arnoldi], Error: (zc)naupd_ IDO = %d\n", ido);
        }
      }
      if (info < 0) {
        clean_arpack_complex_buffer(resid, v, workd, workl, rwork, select, d, z, workev);
        cytnx_error_msg(true, "[ERROR][Arnoldi], Error: (zc)naupd_ INFO = %d\n", info);
      } else if (info == 1) {
        if (verbose) {
          cytnx_warning_msg(true,
                            "[WARNING][Anoldi]: Maximum number of iterations reached."
                            "Try to increase the maximum number of iteratoin.\n"
                            "z(c)naupd info = %d\n",
                            info);
        }
      } else if (info == 3) {
        clean_arpack_complex_buffer(resid, v, workd, workl, rwork, select, d, z, workev);
        cytnx_error_msg(true,
                        "[ERROR][Arnoldi], No shifts could be applied during a cycle of "
                        "the Implicitly Restarted Arnoldi process\n ."
                        "z(c)naupd info = %d\n",
                        info);
      }
      /// calculate eigenvalue and eigenvector by zneupd
      cytnx_int32 rvec = static_cast<cytnx_int32>(is_V);  /// 0: only eigenvalue, 1: also
                                                          /// eigenvector
      char howmny =
        'A';  /// how many eigenvectors to calculate: 'A' => nev eigenvectors
              ///  when howmny == 'A', this is used as workspace to reorder the eigenvectors
      func_xneupd(&rvec, &howmny, select, d, z, &ldv, &sigma, workev, &bmat, &dim, which, &nev,
                  &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, rwork, &info);
      if (info != 0) {
        clean_arpack_complex_buffer(resid, v, workd, workl, rwork, select, d, z, workev);
        cytnx_error_msg(true, "[ERROR][Arnoldi], Error: (zc)neupd_ INFO = %d\n", info);
      }

      auto sorted_idx = sort_metric_indices<T, T2>(d, which, k);
      T* eigvals_data = get_obj_data_ptr<T, T_ten>(out[0]);
      for (cytnx_int32 ik = 0; ik < k; ++ik) {
        eigvals_data[ik] = d[sorted_idx[ik]];
      }

      if (is_V) {
        T* z_data_ptr = reinterpret_cast<T*>(z);
        if constexpr (std::is_same_v<T_ten, UniTensor>) {
          for (cytnx_int32 ik = 0; ik < k; ++ik) {
            T* z_k_ptr = z_data_ptr + sorted_idx[ik] * dim;
            pass_data_UT<T, T_ten>(out[ik + 1], z_k_ptr, true);
          }
        } else if constexpr (std::is_same_v<T_ten, Tensor>) {
          T* tens_data = get_obj_data_ptr<T, T_ten>(out[1]);
          for (cytnx_int32 ik = 0; ik < k; ++ik) {
            T* tmp_data = tens_data + ik * dim;
            T* z_k_ptr = z_data_ptr + sorted_idx[ik] * dim;
            if (Hop->device() == Device.cpu) {
              memcpy(tmp_data, z_k_ptr, dim * sizeof(T));
            } else {
  #ifdef UNI_GPU
              checkCudaErrors(
                cudaMemcpy(tmp_data, z_k_ptr, dim * sizeof(T), cudaMemcpyHostToDevice));
  #endif
            }
          }
        }
      }

      clean_arpack_complex_buffer(resid, v, workd, workl, rwork, select, d, z, workev);
    }

    template <typename T>
    static void clean_arpack_real_buffer(T*& resid, T*& v, T*& workd, T*& workl, T*& dr, T*& di,
                                         cytnx_int32*& select, T*& z, T*& workev) {
      delete[] v;
      v = nullptr;
      delete[] resid;
      resid = nullptr;
      delete[] workd;
      workd = nullptr;
      delete[] workl;
      workl = nullptr;
      delete[] dr;
      dr = nullptr;
      delete[] di;
      di = nullptr;
      delete[] select;
      select = nullptr;
      delete[] z;
      z = nullptr;
      delete[] workev;
      workev = nullptr;
    }

    // T_ten: Tensor or UniTensor
    // T: cytnx_double or cytnx_float
    template <typename T, typename T_ten>
    void _Arnoldi_internal_real(std::vector<T_ten>& out, LinOp* Hop, const T_ten& UT_init,
                                const std::string which_str, const cytnx_uint64& maxiter,
                                const cytnx_double& CvgCrit, const cytnx_uint64& k,
                                const cytnx_bool& is_V, const cytnx_int32& ncv_in,
                                const cytnx_bool& verbose) {
      std::function<void(cytnx_int32 * ido, char* bmat, cytnx_int32* n, char* which,
                         cytnx_int32* nev, T* tol, T* resid, cytnx_int32* ncv, T* v,
                         cytnx_int32* ldv, cytnx_int32* iparam, cytnx_int32* ipntr, T* workd,
                         T* workl, cytnx_int32* lworkl, cytnx_int32* info)>
        func_xnaupd;
      std::function<void(
        cytnx_int32 * rvec, char* howmny, cytnx_int32* select, T* dr, T* di, T* z, cytnx_int32* ldz,
        T* sigmar, T* sigmai, T* workev, char* bmat, cytnx_int32* n, char* which, cytnx_int32* nev,
        T* tol, T* resid, cytnx_int32* ncv, T* v, cytnx_int32* ldv, cytnx_int32* iparam,
        cytnx_int32* ipntr, T* workd, T* workl, cytnx_int32* lworkl, cytnx_int32* info)>
        func_xneupd;
      auto dtype = Hop->dtype();
      if constexpr (std::is_same_v<T, cytnx_double>) {
        func_xnaupd = arpack::dnaupd_;
        func_xneupd = arpack::dneupd_;
      } else if constexpr (std::is_same_v<T, cytnx_float>) {
        func_xnaupd = arpack::snaupd_;
        func_xneupd = arpack::sneupd_;
      } else {
        static_assert(std::is_same_v<T, cytnx_double> || std::is_same_v<T, cytnx_float>,
                      "Unsupported template types for _Arnoldi_internal_real. T must be "
                      "double / flot.");
      }
      cytnx_int32 dim = Hop->nx();
      cytnx_int32 nev = k;
      cytnx_int32 ido = 0;  /// reverse communication parameter, must be zero before iteration
      char bmat = 'I';  ///'I': standard eigenproblem, 'G': generalized eigenproblem
      char which[2];
      std::memcpy(which, which_str.data(), 2);  /// type of asked eigenvalues
      cytnx_int32 info = 0;
      T tol = static_cast<T>(CvgCrit);
      cytnx_int32 iparam[11];
      iparam[0] = 1;  /// Specifies the shift strategy (1->exact)
      iparam[2] = maxiter;  /// Maximum number of iterations
      iparam[3] = 1;  // current code only for 1
      iparam[6] = 1;  /// Sets the mode of z(c)naupd.
      cytnx_int32 ipntr[14];
      T* resid = new T[dim];
      T_ten buffer_UT = UT_init.clone();
      cytnx_bool ifinit = true;  // not allow for false, currently
      if (ifinit) {
        info = 1;
        pass_data_UT<T, T_ten>(buffer_UT, resid, false);
      } else {
        info = 0;
      }
      // If info = 0, a randomly initial residual vector is used.
      // If info = 1, resid contains the initial guess vector provided by user or from previous run
      // on output, resid contains the final residual vector

      /// the number of Ritz vector, nev+2 <= ncv <= dim
      cytnx_int32 ncv = ncv_in == 0 ? std::min(dim, 2 * nev + 10) : ncv_in;
      cytnx_int32 ldv = dim;  /// leading dimension of v
      T* v = new T[dim * ncv];
      T* workd = new T[3 * dim + 1];
      cytnx_int32 lworkl = 3 * ncv * (ncv + 2);  /// LWORKL must be at least 3*NCV**2 + 6*NCV .
      T* workl = new T[lworkl];
      cytnx_int32* select = new cytnx_int32[ncv];
      T* z = new T[dim * (nev + 1)];
      T* workev = new T[3 * ncv];

      T sigmar, sigmai;
      T* dr = new T[nev + 1];
      T* di = new T[nev + 1];

      /// start iteration
      while (true) {
        func_xnaupd(&ido, &bmat, &dim, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr,
                    workd, workl, &lworkl, &info);
        if (ido == -1 || ido == 1) {
          matvec(Hop, buffer_UT, &workd[ipntr[0] - 1], &workd[ipntr[1] - 1]);
        } else if (ido == 99) {
          break;
        } else {
          clean_arpack_real_buffer(resid, v, workd, workl, dr, di, select, z, workev);
          cytnx_error_msg(true, "[ERROR][Arnoldi], Error: (zc)naupd_ IDO = %d\n", ido);
        }
      }
      if (info < 0) {
        clean_arpack_real_buffer(resid, v, workd, workl, dr, di, select, z, workev);
        cytnx_error_msg(true, "[ERROR][Arnoldi], Error: (zc)naupd_ INFO = %d\n", info);
      } else if (info == 1) {
        if (verbose) {
          cytnx_warning_msg(true,
                            "[WARNING][Anoldi]: Maximum number of iterations reached."
                            "Try to increase the maximum number of iteratoin.\n"
                            "z(c)naupd info = %d\n",
                            info);
        }
      } else if (info == 3) {
        clean_arpack_real_buffer(resid, v, workd, workl, dr, di, select, z, workev);
        cytnx_error_msg(true,
                        "[ERROR][Arnoldi], No shifts could be applied during a cycle of "
                        "the Implicitly Restarted Arnoldi process\n ."
                        "z(c)naupd info = %d\n",
                        info);
      }
      /// calculate eigenvalue and eigenvector by dneupd
      cytnx_int32 rvec = static_cast<cytnx_int32>(is_V);  /// 0: only eigenvalue, 1: also
                                                          /// eigenvector
      char howmny =
        'A';  /// how many eigenvectors to calculate: 'A' => nev eigenvectors
              ///  when howmny == 'A', this is used as workspace to reorder the eigenvectors
      func_xneupd(&rvec, &howmny, select, dr, di, z, &ldv, &sigmar, &sigmai, workev, &bmat, &dim,
                  which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl,
                  &info);
      if (info != 0) {
        clean_arpack_real_buffer(resid, v, workd, workl, dr, di, select, z, workev);
        cytnx_error_msg(true, "[ERROR][Arnoldi], Error: (zc)neupd_ INFO = %d\n", info);
      }

      // post process, pass dr*, di*, z* to object
      T* dr_ptr = dr;
      T* di_ptr = dr;
      T* zr_ptr = z;

      // note that the lenght of dr* and di* is k + 1, need to consider all elem to sort the index.
      cytnx_int32 nconv = iparam[4];  // nconv = k or k + 1, depend on how many real eivenvalue
      std::vector<std::complex<T>> d(nconv, T(0));
      for (cytnx_int32 ik = 0; ik < nconv; ++ik) {
        d[ik].real(dr[ik]);
        d[ik].imag(di[ik]);
        std::cout << dr[ik] << "," << di[ik] << std::endl;
      }
      auto sorted_idx = sort_metric_indices<std::complex<T>, T>(d.data(), which, nconv);
      std::complex<T>* eigvals = get_obj_data_ptr<std::complex<T>, T_ten>(out[0]);
      for (cytnx_int32 ik = 0; ik < k; ++ik) {
        eigvals[ik] = d[sorted_idx[ik]];
      }
      if (is_V) {
        const T img_tol = std::numeric_limits<T>::epsilon() * 100;
        T* z_data_ptr = reinterpret_cast<T*>(z);
        auto z_ptr_shifts = std::vector<cytnx_int64>(nconv, 0);
        cytnx_int64 tmp = 0;
        z_ptr_shifts.at(0) = tmp;
        for (cytnx_int32 ik = 0; ik < nconv - 1; ++ik) {
          if (di[ik] <= img_tol) {
            tmp += dim;
            z_ptr_shifts.at(ik + 1) = tmp;
          } else {
            z_ptr_shifts.at(ik + 1) = tmp;
            tmp += 2 * dim;
            if (ik + 2 < nconv) {
              z_ptr_shifts.at(ik + 2) = tmp;
            }
            ik += 1;
          }
        }
        if constexpr (std::is_same_v<T_ten, UniTensor>) {
          if (UT_init.uten_type() == UTenType.Block) {
            for (cytnx_int32 ik = 0; ik < k; ++ik) {
              bool img_positive = di[sorted_idx[ik]] > 0;
              T* zr_k_ptr = zr_ptr + z_ptr_shifts[sorted_idx[ik]];
              // dim is the total element number of a UniTensor contain all blocks
              // block_dim is the element number of a block
              T* zi_k_ptr = zr_k_ptr + dim;
              for (auto& block : out[ik + 1].get_blocks_()) {
                std::complex<T>* eigTen_ptr = get_obj_data_ptr<std::complex<T>, Tensor>(block);
                auto block_dim = get_dim(block);
                // real
                for (int i = 0; i < block_dim; i++) {
                  eigTen_ptr[i].real(*zr_k_ptr++);
                }
                // imag
                if (abs(di[sorted_idx[ik]]) < img_tol) {
                  for (int i = 0; i < block_dim; i++) {
                    eigTen_ptr[i].imag(T(0));
                  }
                } else {
                  for (int i = 0; i < block_dim; i++) {
                    eigTen_ptr[i].imag(img_positive ? (*zi_k_ptr++) : -(*zi_k_ptr++));
                  }
                }
              }
            }
          } else if (UT_init.uten_type() == UTenType.Dense) {
            for (cytnx_int32 ik = 0; ik < k; ++ik) {
              bool img_positive = di[sorted_idx[ik]] > 0;
              std::complex<T>* eigTen_ptr =
                get_obj_data_ptr<std::complex<T>, UniTensor>(out[ik + 1]);
              T* zr_k_ptr = zr_ptr + z_ptr_shifts[sorted_idx[ik]];
              T* zi_k_ptr = zr_k_ptr + dim;
              // real
              for (int i = 0; i < dim; i++) {
                eigTen_ptr[i].real(*zr_k_ptr++);
              }
              // imag
              if (abs(di[sorted_idx[ik]]) < img_tol) {
                for (int i = 0; i < dim; i++) {
                  eigTen_ptr[i].imag(T(0));
                }
              } else {
                for (int i = 0; i < dim; i++) {
                  eigTen_ptr[i].imag(img_positive ? (*zi_k_ptr++) : -(*zi_k_ptr++));
                }
              }
            }
          }
        } else if constexpr (std::is_same_v<T_ten, Tensor>) {
          std::complex<T>* eigTen_ptr = get_obj_data_ptr<std::complex<T>, Tensor>(out[1]);
          for (cytnx_int32 ik = 0; ik < k; ++ik) {
            bool img_positive = di[sorted_idx[ik]] > 0;
            T* zr_k_ptr = zr_ptr + z_ptr_shifts[sorted_idx[ik]];
            T* zi_k_ptr = zr_k_ptr + dim;
            // real
            for (int i = 0; i < dim; i++) {
              eigTen_ptr[i + ik * dim].real(*zr_k_ptr++);
            }
            // imag
            if (abs(di[sorted_idx[ik]]) < img_tol) {
              for (int i = 0; i < dim; i++) {
                eigTen_ptr[i + ik * dim].imag(T(0));
              }
            } else {
              for (int i = 0; i < dim; i++) {
                eigTen_ptr[i + ik * dim].imag(img_positive ? (*zi_k_ptr++) : -(*zi_k_ptr++));
              }
            }
          }
        } else {
        }  // never
      }

      clean_arpack_real_buffer(resid, v, workd, workl, dr, di, select, z, workev);
    }

    void _Arnoldi(std::vector<UniTensor>& out, LinOp* Hop, const UniTensor& UT_init,
                  const std::string which, const cytnx_uint64& maxiter, const double& CvgCrit,
                  const cytnx_uint64& k, const bool& is_V, const cytnx_int32& ncv,
                  const bool& verbose) {
      auto dtype = Hop->dtype();
      auto device = Hop->device();
      auto out_dtype = dtype;
      switch (dtype) {
        case Type.ComplexDouble:
          out_dtype = dtype;
          break;
        case Type.ComplexFloat:
          out_dtype = dtype;
          break;
        case Type.Double:
          out_dtype = Type.ComplexDouble;
          break;
        case Type.Float:
          out_dtype = Type.ComplexFloat;
          break;
      }
      auto eigvals_tens = zeros({k}, out_dtype, device);
      out.push_back(UniTensor(eigvals_tens));
      if (is_V) {
        auto labels = UT_init.labels();
        for (cytnx_int32 ik = 0; ik < k; ++ik) {
          out.push_back(UT_init.clone().astype(out_dtype));
        }
      }

      switch (dtype) {
        case Type.ComplexDouble:
          _Arnoldi_internal_complex<cytnx_complex128, cytnx_double, UniTensor>(
            out, Hop, UT_init, which, maxiter, CvgCrit, k, is_V, ncv, verbose);
          break;
        case Type.ComplexFloat:
          _Arnoldi_internal_complex<cytnx_complex64, cytnx_float, UniTensor>(
            out, Hop, UT_init, which, maxiter, CvgCrit, k, is_V, ncv, verbose);
          break;
        case Type.Double:
          _Arnoldi_internal_real<cytnx_double, UniTensor>(out, Hop, UT_init, which, maxiter,
                                                          CvgCrit, k, is_V, ncv, verbose);
          break;
        case Type.Float:
          _Arnoldi_internal_real<cytnx_float, UniTensor>(out, Hop, UT_init, which, maxiter, CvgCrit,
                                                         k, is_V, ncv, verbose);
          break;
      }
    }

    void _Arnoldi(std::vector<Tensor>& out, LinOp* Hop, const Tensor& UT_init,
                  const std::string which, const cytnx_uint64& maxiter, const double& CvgCrit,
                  const cytnx_uint64& k, const bool& is_V, const cytnx_int32& ncv,
                  const bool& verbose) {
      auto dtype = Hop->dtype();
      auto device = Hop->device();
      auto out_dtype = dtype;
      switch (dtype) {
        case Type.ComplexDouble:
          out_dtype = dtype;
          break;
        case Type.ComplexFloat:
          out_dtype = dtype;
          break;
        case Type.Double:
          out_dtype = Type.ComplexDouble;
          break;
        case Type.Float:
          out_dtype = Type.ComplexFloat;
          break;
      }
      auto eigvals_tens = zeros({k}, out_dtype, device);
      auto dim = Hop->nx();
      out[0] = eigvals_tens;
      if (is_V) {
        auto eigTens =
          k == 1 ? zeros({dim}, out_dtype, device) : zeros({k, dim}, out_dtype, device);
        out[1] = eigTens;
      }

      switch (dtype) {
        case Type.ComplexDouble:
          _Arnoldi_internal_complex<cytnx_complex128, cytnx_double, Tensor>(
            out, Hop, UT_init, which, maxiter, CvgCrit, k, is_V, ncv, verbose);
          break;
        case Type.ComplexFloat:
          _Arnoldi_internal_complex<cytnx_complex64, cytnx_float, Tensor>(
            out, Hop, UT_init, which, maxiter, CvgCrit, k, is_V, ncv, verbose);
          break;
        case Type.Double:
          _Arnoldi_internal_real<cytnx_double, Tensor>(out, Hop, UT_init, which, maxiter, CvgCrit,
                                                       k, is_V, ncv, verbose);
          break;
        case Type.Float:
          _Arnoldi_internal_real<cytnx_float, Tensor>(out, Hop, UT_init, which, maxiter, CvgCrit, k,
                                                      is_V, ncv, verbose);
          break;
      }
    }

    std::vector<Tensor> Arnoldi(LinOp* Hop, const Tensor& T_init, const std::string which,
                                const cytnx_uint64& maxiter, const double& cvg_crit,
                                const cytnx_uint64& k, const bool& is_V, const int& ncv,
                                const bool& verbose) {
      // check type:
      cytnx_error_msg(
        !Type.is_float(Hop->dtype()),
        "[ERROR][Arnoldi] Arnoldi can only accept operator with floating types (complex/real)%s",
        "\n");

      // check which
      std::vector<std::string> accept_which = {"LM", "LR", "LI", "SR", "SI"};
      if (std::find(accept_which.begin(), accept_which.end(), which) == accept_which.end()) {
        cytnx_error_msg(true,
                        "[ERROR][Arnoldi] 'which' should be 'LM', 'LR, 'LI'"
                        ", 'SR', 'SI'",
                        "\n");
      }

      /// check k
      cytnx_error_msg(k < 1, "[ERROR][Arnoldi] k should be >0%s", "\n");
      cytnx_error_msg(k > Hop->nx(),
                      "[ERROR][Arnoldi] k can only be up to total dimension of input vector D%s",
                      "\n");

      // check Tin should be rank-1:
      auto _T_init = T_init.clone();
      if (T_init.dtype() == Type.Void) {
        _T_init =
          cytnx::random::normal({Hop->nx()}, Hop->dtype(), Hop->device());  // randomly initialize.
      } else {
        cytnx_error_msg(T_init.shape().size() != 1, "[ERROR][Arnoldi] Tin should be rank-1%s",
                        "\n");
        cytnx_error_msg(T_init.shape()[0] != Hop->nx(),
                        "[ERROR][Arnoldi] Tin should have dimension consistent with Hop: [%d] %s",
                        Hop->nx(), "\n");
        cytnx_error_msg(
          T_init.dtype() != Hop->dtype(),
          "[ERROR][Arnoldi] Tin should have same data type consistent with Hop: [%d] %s",
          Hop->dtype(), "\n");
      }

      cytnx_error_msg(cvg_crit < 0, "[ERROR][Arnoldi] cvg_crit should be >= 0%s", "\n");
      cytnx_error_msg((ncv != 0) && ((ncv < 2 + k) || ncv > Hop->nx()),
                      "[ERROR][Arnoldi] ncv should "
                      "be 2+k<=ncv<=nx%s",
                      "\n");
      cytnx_uint64 output_size = is_V ? 2 : 1;
      auto out = std::vector<Tensor>(output_size, Tensor());
      _Arnoldi(out, Hop, _T_init, which, maxiter, cvg_crit, k, is_V, ncv, verbose);
      return out;
    }

    std::vector<UniTensor> Arnoldi(LinOp* Hop, const UniTensor& UT_init, const std::string which,
                                   const cytnx_uint64& maxiter, const double& cvg_crit,
                                   const cytnx_uint64& k, const bool& is_V, const int& ncv,
                                   const bool& verbose) {
      // check type:
      cytnx_error_msg(
        !Type.is_float(Hop->dtype()),
        "[ERROR][Arnoldi] Arnoldi can only accept operator with floating types (complex/real)%s",
        "\n");

      // check which
      std::vector<std::string> accept_which = {"LM", "LR", "LI", "SR", "SI"};
      if (std::find(accept_which.begin(), accept_which.end(), which) == accept_which.end()) {
        cytnx_error_msg(true,
                        "[ERROR][Arnoldi] 'which' should be 'LM', 'LR, 'LI'"
                        ", 'SR', 'SI'",
                        "\n");
      }

      /// check k
      cytnx_error_msg(k < 1, "[ERROR][Arnoldi] k should be >0%s", "\n");
      cytnx_error_msg(k > Hop->nx(),
                      "[ERROR][Arnoldi] k can only be up to total dimension of input vector D%s",
                      "\n");

      // check Tin should be rank-1:
      if (UT_init.dtype() == Type.Void) {
        cytnx_error_msg(k < 1, "[ERROR][Arnoldi] The initial UniTensor sould be defined.%s", "\n");
      } else {
        auto dim = get_elem_num(UT_init);
        cytnx_error_msg(dim != Hop->nx(),
                        "[ERROR][Arnoldi] Tin should have dimension consistent with Hop: [%d] %s",
                        Hop->nx(), "\n");
        cytnx_error_msg(UT_init.dtype() != Hop->dtype(),
                        "[ERROR][Arnoldi] Tin should have datatype consistent with Hop: [%d] %s",
                        Hop->dtype(), "\n");
      }

      cytnx_error_msg(cvg_crit < 0, "[ERROR][Arnoldi] cvg_crit should be >= 0%s", "\n");
      cytnx_error_msg((ncv != 0) && ((ncv < 2 + k) || ncv > Hop->nx()),
                      "[ERROR][Arnoldi] ncv should "
                      "be 2+k<=ncv<=nx%s",
                      "\n");
      auto out = std::vector<UniTensor>();
      _Arnoldi(out, Hop, UT_init, which, maxiter, cvg_crit, k, is_V, ncv, verbose);
      return out;
    }

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
