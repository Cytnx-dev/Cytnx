#include "linalg.hpp"
#include "Generator.hpp"
#include "random.hpp"
#include "Tensor.hpp"
#include "LinOp.hpp"

#include <cfloat>
#include <algorithm>
namespace cytnx {
  namespace linalg {
    typedef Accessor ac;

    //resize the matrix (2-rank tensor)
    static UniTensor ResizeMat_Ut(const UniTensor& src, const cytnx_uint64 r, const cytnx_uint64 c)
    {
      const auto min_r = std::min(r, src.shape()[0]);
      const auto min_c = std::min(c, src.shape()[1]);
      Tensor dst = cytnx::zeros({min_r, min_c}, src.dtype(), src.device());
      for(cytnx_uint64 i = 0; i < min_r; ++i)
      {
        for(cytnx_uint64 j = 0; j < min_c; ++j) {
          dst[{i, j}] = src[{i, j}];
        }
      }
      return dst;
    }

    //Get the indices of the first few order element
    std::vector<cytnx_int32> GetFstFewOrderElemIdices(
        const Tensor& tens, const std::string& which, const cytnx_int64 k)
    {
      char large_or_small = which[0]; //'S' or 'L'
      char metric_type = which[1]; //'M', 'R' or 'I'

      //get the metric distance
      auto len = tens.shape()[0];
      std::vector<Scalar> vec;
      vec = std::vector<Scalar>(len, 0);
      if(metric_type == 'M') {
        for(int i = 0; i < len; ++i) 
          vec[i] = abs(tens.storage().at(i));
      } else if(metric_type == 'R') {
        for(int i = 0; i < len; ++i) 
          vec[i] = tens.storage().at(i).real();
      } else if(metric_type == 'I') {
        for(int i = 0; i < len; ++i) 
          vec[i] = tens.storage().at(i).imag();
      } else {;} //never
      
      //smallest or largest
      bool is_small = (large_or_small == 'S');
      Scalar init_scalar = is_small ? Scalar::maxval(Type.Double) : 0;
      auto indices = std::vector<cytnx_int32>(k, -1);
      for(cytnx_int32 i = 0; i < k; ++i)
      {
        auto itr = is_small ? 
	  std::min_element(vec.begin(), vec.end()) :
	  std::max_element(vec.begin(), vec.end());
	indices[i] = static_cast<cytnx_int32>(itr - vec.begin());
	*itr = init_scalar;
      }
      return indices;
    }

    bool IsEigvalCvg_Ut(const std::vector<Scalar> &eigvals, 
	    const std::vector<Scalar> &eigvals_old, const double cvg_crit)
    {
      for(cytnx_int32 i = 0; i < eigvals.size(); ++i)
      {
        auto err = abs(eigvals[i] - eigvals_old[i]);
	if(err >= cvg_crit)
	  return false;
      }
      return true;
    }

    //check the residule |Mv - ev| is converged.
    bool IsResiduleSmallEnough_Ut(LinOp *Hop, const std::vector<UniTensor> &eigvecs, 
	    const std::vector<Scalar> &eigvals, const double cvg_crit)
    {
      for(cytnx_int32 i = 0; i < eigvals.size(); ++i)
      {
	auto eigvec = eigvecs[i];
	auto eigval = eigvals[i];
	auto resi = (Hop->matvec(eigvec) - eigval * eigvec).Norm().item();
	if(resi >= cvg_crit)
	  return false;
      }
      return true;
    }

    std::vector<UniTensor> GetEigTens_Ut(const std::vector<UniTensor> qs, 
		    Tensor eigvec_in_kryv, std::vector<cytnx_int32> max_indices)
    {
      auto k = max_indices.size();
      cytnx_int64 krydim = eigvec_in_kryv.shape()[0];
      auto P_inv = InvM(eigvec_in_kryv).Conj();
      auto eigTens_s = std::vector<UniTensor> (k, UniTensor());
      for(cytnx_int32 ik = 0; ik < k; ++ik)
      { 
	auto maxIdx = max_indices[ik];
        auto eigTens = zeros(qs[0].shape(), Type.ComplexDouble);
        for(cytnx_int64 i = 0; i < krydim; ++i)
        {
          eigTens += P_inv[{i, maxIdx}] * qs[i];
        }
	eigTens /= eigTens.Norm().item();
	eigTens_s[ik] = UniTensor(eigTens);
      }
      return eigTens_s;
    }

    void _Arnoldi_Ut(std::vector<UniTensor>& out, LinOp *Hop, 
		 const UniTensor& T_init, 
                 const std::string which, 
		 const cytnx_uint64 &maxiter, 
		 const double &CvgCrit, 
		 const cytnx_uint64 &k, 
		 const bool &is_V, const bool &verbose) {
      auto vec_len = T_init.shape()[0];
      const cytnx_uint64 imp_maxiter = std::min(maxiter, vec_len + 1);
      const cytnx_complex128 unit_complex = 1.0;
      // out[0]:eigenvalues, out[1]:eigentensors
      out[0] = UniTensor(zeros({k}, Type.ComplexDouble)); // initialize
      std::vector<Scalar> eigvals = std::vector<Scalar>(k, Scalar());
      std::vector<UniTensor> eigTens_s;
      UniTensor kry_mat_buffer = UniTensor(cytnx::zeros({imp_maxiter + 1, imp_maxiter + 1}, Hop->dtype(), Hop->device()));
      bool is_cvg = false;
      std::vector<Scalar> eigvals_old = std::vector<Scalar>(k, Scalar::maxval(Type.Double));
      std::vector<UniTensor> buffer;
      buffer.push_back(T_init);
      buffer[0] = buffer[0] / buffer[0].Norm().item();  // normalized q1

      //start arnoldi iteration
      for (cytnx_uint64 i = 1; i < imp_maxiter; i++) {
        cytnx_uint64 krydim = i;
        UniTensor nextTens = Hop->matvec(buffer[i - 1]).astype(Hop->dtype());
        buffer.push_back(nextTens);
        for (cytnx_uint32 j = 0; j < krydim; j++) {
          // auto h = Vectordot(buffer[i], buffer[j], true).Conj();
          UniTensor h = buffer[i].contract(buffer[j], true, true).Conj();
          kry_mat_buffer.at({i - 1, j}) = h.item();
          buffer[i] -= h * buffer[j];
        }
        auto h = buffer[i].Norm().item();
        kry_mat_buffer.at({i - 1, i}) = h;
        // kry_mat_buffer[{i - 1, i}] = h;
        buffer[i] /= h;
        UniTensor kry_mat = ResizeMat_Ut(kry_mat_buffer, krydim, krydim);

	//call Eig to get eigenvalues
        std::vector<Tensor> eigs = Eig(kry_mat.get_block_(), true, true);
	// get first few order of eigenvlues
        std::vector<cytnx_int32> maxIndices = GetFstFewOrderElemIdices(eigs[0], which, k);
        for(cytnx_int32 ik = 0; ik < k; ++ik)
        {
          auto maxIdx = maxIndices[ik];
          eigvals[ik] = eigs[0].storage().at(maxIdx);
        }

	//check converged
        bool is_eigval_cvg = IsEigvalCvg_Ut(eigvals, eigvals_old, CvgCrit);
        if(is_eigval_cvg || i == imp_maxiter - 1) {
          eigTens_s = GetEigTens_Ut(buffer, eigs[1], maxIndices);
          bool is_res_small_enough = 
              IsResiduleSmallEnough_Ut(Hop, eigTens_s, eigvals, CvgCrit);
          if(is_res_small_enough)
          {
            is_cvg = true;
            break;
          }
        }
        eigvals_old = eigvals;
      } // Arnoldi iteration
      buffer.clear();

      // set output
      for(cytnx_int32 ik = 0; ik < k; ++ik)
        // out[0][{ac(ik)}] = eigvals[ik];
        out[0].at({ik}) = eigvals[ik];
      if (is_V) // if need output eigentensors
      {
	out[1] = eigTens_s[0];
        if (eigTens_s.size() > 1) // if k > 1, append the eigenvector as a single Tensor'.
        {
          out[1].reshape_({1, -1});
	  for(cytnx_uint64 i = 1; i < eigTens_s.size(); ++i)
            out[1].append(eigTens_s[i]);
	}
      }
    }

    std::vector<UniTensor> Arnoldi(LinOp *Hop, const UniTensor& T_init, 
		                const std::string which, 
				const cytnx_uint64 &maxiter, 
				const double &cvg_crit, 
				const cytnx_uint64 &k, 
				const bool &is_V, 
				const bool &verbose) {
      // check type:
      cytnx_error_msg(
        !Type.is_float(Hop->dtype()),
        "[ERROR][Lanczos] Lanczos can only accept operator with floating types (complex/real)%s",
        "\n");

      //check which
      std::vector<std::string> accept_which = {"LM", "LR", "LI", "SM", "SR", "SI"};
      if (std::find(accept_which.begin(), accept_which.end(), which) ==  accept_which.end()) {
        cytnx_error_msg(true, "[ERROR][Lanczos] 'which' should be 'LM', 'LR, 'LI'"
			", 'SM', 'SR, 'SI'", "\n");
      }

      /// check k
      cytnx_error_msg(k < 1, "[ERROR][Lanczos] k should be >0%s", "\n");
      cytnx_error_msg(k > Hop->nx(),
                      "[ERROR][Lanczos] k can only be up to total dimension of input vector D%s",
                      "\n");

      // check Tin should be rank-1:
      auto _T_init = T_init.clone();
      _T_init = T_init.astype(Hop->dtype());

      // if (T_init.dtype() == Type.Void) {
      //   _T_init =
      //     cytnx::random::normal({Hop->nx()}, Hop->dtype(), Hop->device());  // randomly initialize.
      // } else {
      //   cytnx_error_msg(T_init.shape().size() != 1, "[ERROR][Lanczos] Tin should be rank-1%s", "\n");
      //   cytnx_error_msg(T_init.shape()[0] != Hop->nx(),
      //                   "[ERROR][Lanczos] Tin should have dimension consistent with Hop: [%d] %s",
      //                   Hop->nx(), "\n");
      //   _T_init = T_init.astype(Hop->dtype());
      // }

      cytnx_error_msg(cvg_crit <= 0, "[ERROR][Lanczos] cvg_crit should be > 0%s", "\n");
      double _cvgcrit = cvg_crit;
      cytnx_uint64 output_size = is_V ? 2 : 1;
      auto out = std::vector<UniTensor>(output_size, UniTensor());
      _Arnoldi_Ut(out, Hop, T_init, which, maxiter, cvg_crit, k, is_V, verbose);
      return out;
    }
  }  // namespace linalg
}  // namespace cytnx
