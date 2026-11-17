#ifndef CYTNX_BACKEND_DECOMPOSITION_ALGORITHM_H_
#define CYTNX_BACKEND_DECOMPOSITION_ALGORITHM_H_

#include <string>
#include <string_view>

namespace cytnx {
  namespace linalg_internal {

    // The values of the `algorithm` option of the dense decompositions. This is the only place
    // that spells them, so that Eigh (#1214) and, later, Svd (#623) share one vocabulary.
    inline constexpr std::string_view kAlgorithmDefault = "default";
    // QR iteration: ?syev / ?heev.
    inline constexpr std::string_view kAlgorithmQr = "qr";
    // Divide and conquer: ?syevd / ?heevd, cusolverDn?syevd / cusolverDn?heevd.
    inline constexpr std::string_view kAlgorithmDc = "dc";

    // The driver of the symmetric / Hermitian eigenproblem that a kernel runs.
    enum class EighDriver { kQr, kDc };

    // Maps the `algorithm` option of Eigh to a driver, for a tensor on \p device. This is the
    // only place that decides what "default" means. Raises cytnx::error for a name that is
    // unknown, and for a driver that the backend of \p device does not have.
    EighDriver resolve_eigh_algorithm(const std::string &algorithm, int device);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_DECOMPOSITION_ALGORITHM_H_
