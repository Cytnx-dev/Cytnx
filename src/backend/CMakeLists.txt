

######################################################################
### ADD SUB-DIRECTORIES
######################################################################

target_sources_local(cytnx
  PRIVATE
    linalg_internal_interface.hpp
    linalg_internal_interface.cpp

    random_internal_interface.hpp
    random_internal_interface.cpp

    algo_internal_interface.hpp
    algo_internal_interface.cpp

    utils_internal_interface.hpp
    utils_internal_interface.cpp

    Scalar.cpp
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

    Tensor_impl.cpp
)


add_subdirectory(linalg_internal_cpu)
add_subdirectory(random_internal_cpu)
add_subdirectory(algo_internal_cpu)
add_subdirectory(utils_internal_cpu)
if(USE_CUDA)
  add_subdirectory(linalg_internal_gpu)
  add_subdirectory(random_internal_gpu)
  add_subdirectory(algo_internal_gpu)
  add_subdirectory(utils_internal_gpu)
endif()




#if(UNI_GPU)
#    target_sourse_local(cytnx
#        PRIVATE
#            Device.cu
#    )
#endif

#add_subdirectory(linalg)
#add_subdirectory(algo)
#add_subdirectory(stat)
#add_subdirectory(utils)
#add_subdirectory(random)
#add_subdirectory(tn_algo)
