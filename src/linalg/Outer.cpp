#include "linalg.hpp"
#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else

namespace cytnx {

  namespace linalg {
    // Outer(a, b) for rank-1 a (length m) and rank-1 b (length n) is the m x n
    // matrix out[i, j] = a_i * b_j. That is exactly the Kronecker product of the
    // two vectors -- Kron(a, b) is a length-(m*n) vector whose element (i*n + j)
    // equals a_i * b_j -- reshaped to {m, n}. Delegating to Kron reuses its
    // variant-based typed dispatch, which covers every dtype pair (including
    // Int16/Uint16/Bool) on both CPU and GPU and shares the same
    // Type.type_promote result dtype. The former per-dtype Outer_ii / cuOuter_ii
    // dispatch tables are therefore retired: their off-diagonal entries were dead
    // (Outer cast both operands to the promoted dtype before dispatching) and
    // their Int16/Uint16/Bool rows were missing, which segfaulted (#1099).
    Tensor Outer(const Tensor &Tl, const Tensor &Tr) {
      cytnx_error_msg(Tl.is_void(), "[ERROR] pass empty tensor in param #1%s", "\n");
      cytnx_error_msg(Tr.is_void(), "[ERROR] pass empty tensor in param #2%s", "\n");
      cytnx_error_msg(Tl.device() != Tr.device(),
                      "[ERROR] two tensor cannot on different devices.%s", "\n");
      cytnx_error_msg(Tl.shape().size() != 1, "[ERROR] tensor #1 should have rank-1.%s", "\n");
      cytnx_error_msg(Tr.shape().size() != 1, "[ERROR] tensor #2 should have rank-1.%s", "\n");

      return Kron(Tl, Tr).reshape(static_cast<cytnx_int64>(Tl.shape()[0]),
                                  static_cast<cytnx_int64>(Tr.shape()[0]));
    }

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
