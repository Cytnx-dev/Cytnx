target_sources_local(cytnx
  PRIVATE
    # cuNorm_internal.hpp
    # cuAbs_internal.hpp
    # cuAdd_internal.hpp
    # cuGer_internal.hpp
    # cuArithmetic_internal.hpp
    # cuConj_inplace_internal.hpp
    # cuCpr_internal.hpp
    # cuDiag_internal.hpp
    # cuDiv_internal.hpp
    # cuPow_internal.hpp
    # cuEigh_internal.hpp
    # cuExp_internal.hpp
    # cuInvM_inplace_internal.hpp
    # cuInv_inplace_internal.hpp
    # cuGemm_internal.hpp
    # cuGemm_Batch_internal.hpp
    # cuMatmul_internal.hpp
    # cuMatmul_dg_internal.hpp
    # cuMatvec_internal.hpp
    # cuVectordot_internal.hpp
    # cuMul_internal.hpp
    # cuOuter_internal.hpp
    # cuSub_internal.hpp
    # cuGeSvd_internal.hpp
    # cuSvd_internal.hpp
    # cuDet_internal.hpp
    # cuSum_internal.hpp
    # cuMaxMin_internal.hpp
    # cuKron_internal.cuh
    # cuTensordot_internal.hpp
    # cuQuantumGeSvd_internal.hpp
    # cuQuantumQr_internal.hpp
    # cudaMemcpyTruncation.hpp

    cuAbs_internal.cu
    cuAdd_internal.cu
    cuGer_internal.cu
    cuArithmetic_internal.cu
    cuConj_inplace_internal.cu
    cuCpr_internal.cu
    cuDiag_internal.cu
    cuDiv_internal.cu
    cuPow_internal.cu
    cuEigh_internal.cu
    cuExp_internal.cu
    cuInvM_inplace_internal.cu
    cuInv_inplace_internal.cu
    cuGemm_internal.cu
    cuGemm_Batch_internal.cu
    cuMatmul_internal.cu
    cuMatmul_dg_internal.cu
    cuMatvec_internal.cu
    cuVectordot_internal.cu
    cuMul_internal.cu
    cuOuter_internal.cu
    cuSub_internal.cu
    cuGeSvd_internal.cu
    cuSvd_internal.cu
    cuNorm_internal.cu
    cuMod_internal.cu
    cuDet_internal.cu
    cuSum_internal.cu
    cuMaxMin_internal.cu
    cuTensordot_internal.cu
    cuQuantumGeSvd_internal.cu
    cuQuantumQr_internal.cu
    cudaMemcpyTruncation.cu
)
