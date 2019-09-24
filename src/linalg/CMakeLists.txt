  target_sources_local(cytnx
    # PUBLIC
    # linalg.hpp
    # linalg_internal_interface.hpp
    PRIVATE
    Add.cpp
    Conj.cpp
    Conj_.cpp
    Cpr.cpp
    Diag.cpp
    Div.cpp
    Eigh.cpp
    Exp.cpp
    Exp_.cpp
    Inv.cpp
    Inv_.cpp
    Matmul.cpp
    Mul.cpp
    Otimes.cpp
    Sub.cpp
    Svd.cpp
    Svd_truncate.cpp
    Tensordot.cpp
    Vectordot.cpp
    linalg_internal_interface.cpp
)

add_subdirectory(linalg_internal_cpu)
if(USE_CUDA)
  add_subdirectory(linalg_internal_gpu)
endif()
