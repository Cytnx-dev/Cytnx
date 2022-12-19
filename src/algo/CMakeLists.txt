target_sources_local(cytnx
    PRIVATE
    algo_internal_interface.hpp
    algo_internal_interface.cpp
    Sort.cpp
    Concatenate.cpp
    Vstack.cpp
)

add_subdirectory(algo_internal_cpu)
if(USE_CUDA)
  add_subdirectory(algo_internal_gpu)
endif()
