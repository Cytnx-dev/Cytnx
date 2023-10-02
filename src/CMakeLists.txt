# #####################################################################
# ## ADD SUB-DIRECTORIES
# #####################################################################

target_sources_local(cytnx
  PRIVATE

  # Scalar.cpp
  Network_base.cpp
  Network.cpp
  RegularNetwork.cpp
  FermionNetwork.cpp
  Bond.cpp
  UniTensor_base.cpp
  UniTensor.cpp
  DenseUniTensor.cpp

  # SparseUniTensor.cpp
  BlockUniTensor.cpp
  Symmetry.cpp
  contraction_tree.cpp
  search_tree.cpp
  Device.cpp
  Generator.cpp
  Physics.cpp
  Accessor.cpp
  LinOp.cpp
  Type.cpp
  Tensor.cpp

  # Tensor_impl.cpp
  ncon.cpp
)

# if(UNI_GPU)
# target_sourse_local(cytnx
# PRIVATE
# Device.cu
# )
# endif
add_subdirectory(linalg)
add_subdirectory(algo)
add_subdirectory(stat)
add_subdirectory(utils)
add_subdirectory(random)
add_subdirectory(tn_algo)

if(BACKEND_TORCH)
  add_subdirectory(backend_torch)
else()
  add_subdirectory(backend)
endif()
