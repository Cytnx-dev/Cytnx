

######################################################################
### ADD SUB-DIRECTORIES
######################################################################

target_sources_local(cytnx
  PRIVATE
    Scalar.cpp
    Network_base.cpp
    Network.cpp
    RegularNetwork.cpp
    FermionNetwork.cpp
    Bond.cpp
    UniTensor_base.cpp
    UniTensor.cpp
    DenseUniTensor.cpp
    SparseUniTensor.cpp
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
    Storage.cpp
    Storage_base.cpp
    ComplexDoubleStorage.cpp
    ComplexFloatStorage.cpp
    DoubleStorage.cpp
    FloatStorage.cpp
    Uint64Storage.cpp
    Int64Storage.cpp
    Uint32Storage.cpp
    Int32Storage.cpp
    Uint16Storage.cpp
    Int16Storage.cpp
    BoolStorage.cpp
    ncon.cpp
)

#if(UNI_GPU)
#    target_sourse_local(cytnx
#        PRIVATE
#            Device.cu
#    )
#endif

add_subdirectory(linalg)
add_subdirectory(algo)
add_subdirectory(stat)
add_subdirectory(utils)
add_subdirectory(random)
add_subdirectory(tn_algo)
