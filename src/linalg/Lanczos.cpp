#include "linalg.hpp"
#include "Generator.hpp"
#include "random.hpp"
#include "Tensor.hpp"
#include "LinOp.hpp"

#include <cfloat>
#include <vector>
#include <cmath>
#include "UniTensor.hpp"
#include "utils/vec_print.hpp"
#include <iomanip>

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  namespace linalg {
    typedef Accessor ac;
    using namespace std;

    vector<Tensor> Lanczos(LinOp *Hop, const Tensor &Tin, const string method,
                           const double &CvgCrit, const unsigned int &Maxiter,
                           const cytnx_uint64 &k, const bool &is_V, const bool &is_row,
                           const cytnx_uint32 &max_krydim, const bool &verbose) {
      if (method.compare("ER") == 0) {
        return Lanczos_ER(Hop, k, is_V, Maxiter, CvgCrit, is_row, Tin, max_krydim, verbose);
      } else if (method.compare("Gnd") == 0) {
        cytnx_error_msg(k > 1, "[ERROR][Lanczos] Only k = 1 is supported for 'Gnd' method.%s",
                        "\n");
        cytnx_warning_msg(
          max_krydim > 0,
          "[WARNING][Lanczos] max_krydim > 0 while it is irrelevent when using 'Gnd' method.%s",
          "\n");
        return Lanczos_Gnd(Hop, CvgCrit, is_V, Tin, verbose, Maxiter);
      } else {
        cytnx_error_msg(
          1, "[ERROR][Lanczos] Invalid Lanczos method, should be either 'ER' or 'Gnd'.%s", "\n");
        return vector<Tensor>();
      }
    }  // Lanczos

    vector<UniTensor> Lanczos(LinOp *Hop, const UniTensor &Tin, const string method,
                              const double &CvgCrit, const unsigned int &Maxiter,
                              const cytnx_uint64 &k, const bool &is_V, const bool &is_row,
                              const cytnx_uint32 &max_krydim, const bool &verbose) {
      if (method.compare("ER") == 0) {
        cytnx_error_msg(
          1, "[ERROR][Lanczos] Lanczos method 'ER' for UniTensor is under developing!.%s", "\n");
        return vector<UniTensor>();
      } else if (method.compare("Gnd") == 0) {
        cytnx_error_msg(k > 1, "[ERROR][Lanczos] Only k = 1 is supported for 'Gnd' method.%s",
                        "\n");
        cytnx_warning_msg(
          max_krydim > 0,
          "[WARNING][Lanczos] max_krydim > 0 while is irrelevent when using 'Gnd' method.%s", "\n");
        return Lanczos_Gnd_Ut(Hop, Tin, CvgCrit, is_V, verbose, Maxiter);
      } else {
        cytnx_error_msg(
          1, "[ERROR][Lanczos] Invalid Lanczos method, should be either 'ER' or 'Gnd'.%s", "\n");
        return vector<UniTensor>();
      }
    }  // Lanczos

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
