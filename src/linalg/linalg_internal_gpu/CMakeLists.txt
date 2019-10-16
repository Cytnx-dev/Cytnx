target_sources_local(cytnx
  PRIVATE
    cuAdd_internal.hpp
    cuArithmetic_internal.hpp
    cuConj_inplace_internal.hpp
    cuCpr_internal.hpp
    cuDiag_internal.hpp
    cuDiv_internal.hpp
    cuEigh_internal.hpp
    cuExp_internal.hpp
    cuInv_inplace_internal.hpp
    cuMatmul_internal.hpp
    cuMul_internal.hpp
    cuOuter_internal.hpp
    cuSub_internal.hpp
    cuSvd_internal.hpp
    cuAdd_internal.cu
    cuArithmetic_internal.cu
    cuConj_inplace_internal.cu
    cuCpr_internal.cu
    cuDiag_internal.cu
    cuDiv_internal.cu
    cuEigh_internal.cu
    cuExp_internal.cu
    cuInv_inplace_internal.cu
    cuMatmul_internal.cu
    cuMul_internal.cu
#    cuOuter_internal.cu
    cuSub_internal.cu
    cuSvd_internal.cu
  )
#CUDA_ADD_LIBRARY(cytnx
#        cuAdd_internal.hpp
#        cuArithmetic_internal.hpp
#        cuConj_inplace_internal.hpp
#        cuCpr_internal.hpp
#        cuDiag_internal.hpp
#        cuDiv_internal.hpp
#        cuEigh_internal.hpp
#        cuExp_internal.hpp
#        cuInv_inplace_internal.hpp
#        cuMatmul_internal.hpp
#        cuMul_internal.hpp
#        cuOuter_internal.hpp
#        cuSub_internal.hpp
#        cuSvd_internal.hpp
#        cuAdd_internal.cu
#        cuArithmetic_internal.cu
#        cuConj_inplace_internal.cu
#        cuCpr_internal.cu
#        cuDiag_internal.cu
#        cuDiv_internal.cu
#        cuEigh_internal.cu
#        cuExp_internal.cu
#        cuInv_inplace_internal.cu
#        cuMatmul_internal.cu
#        cuMul_internal.cu
#        cuOuter_internal.cu
#        cuSub_internal.cu
#        cuSvd_internal.cu
#
#        )
