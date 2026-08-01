# Locate the CUDA 12+/13 `cccl/` include subdir that holds <cuda/std/complex>.
#
# UNI_GPU is applied PUBLIC, so cytnx's public header Type.hpp includes
# <cuda/std/complex>. CUDA 12+/13 relocate that header under the toolkit's
# `cccl/` subdir, which is part of no CUDA:: imported target's INTERFACE. Both
# the in-tree build and an installed find_package(Cytnx) GPU consumer therefore
# need this dir added explicitly. The detection is factored here so both sites
# use identical logic against whatever toolkit is active in their context.
#
# Requires find_package(CUDAToolkit) to have run first (so CUDAToolkit_INCLUDE_DIRS
# and CUDAToolkit_TARGET_DIR are set). Sets <out_var> in the caller's scope to the
# absolute cccl dir, or "" when none exists (e.g. CUDA < 12, where the header is
# already on the default include path).
function(cytnx_detect_cccl_include_dir out_var)
  set(candidates)
  if(DEFINED CUDAToolkit_TARGET_DIR AND NOT "${CUDAToolkit_TARGET_DIR}" STREQUAL "")
    list(APPEND candidates "${CUDAToolkit_TARGET_DIR}/include/cccl")
  endif()
  foreach(cuda_inc IN LISTS CUDAToolkit_INCLUDE_DIRS)
    list(APPEND candidates
      "${cuda_inc}/cccl"
      "${cuda_inc}/../include/cccl"
      "${cuda_inc}/../../include/cccl"
      "${cuda_inc}/../../../include/cccl")
  endforeach()
  list(REMOVE_DUPLICATES candidates)

  set(found "")
  foreach(candidate IN LISTS candidates)
    get_filename_component(candidate_abs "${candidate}" ABSOLUTE)
    if(EXISTS "${candidate_abs}")
      set(found "${candidate_abs}")
      break()
    endif()
  endforeach()
  set(${out_var} "${found}" PARENT_SCOPE)
endfunction()
