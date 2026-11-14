#pragma once
#include <limits>

#include "Type.hpp"

namespace cytnx {
  namespace linalg {

    // Happy-breakdown threshold shared by Lanczos_Gnd and Lanczos_Gnd_Ut. A residual norm beta_i
    // that is not larger than this value times the size of the Lanczos coefficients is roundoff:
    // the Krylov space is then an invariant subspace of the operator, and dividing by beta_i would
    // turn that roundoff into a spurious Krylov vector. The factor of 100 matches the one
    // Lanczos_Exp uses.
    inline double lanczos_gnd_breakdown_tolerance(unsigned int dtype) {
      constexpr double roundoff_factor = 100.0;
      if (dtype == Type.Float || dtype == Type.ComplexFloat) {
        return roundoff_factor * std::numeric_limits<float>::epsilon();
      }
      return roundoff_factor * std::numeric_limits<double>::epsilon();
    }

  }  // namespace linalg
}  // namespace cytnx
