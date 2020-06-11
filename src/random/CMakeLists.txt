target_sources_local(cytnx
    PRIVATE
    random_internal_interface.hpp
    random_internal_interface.cpp
    Make_normal.cpp
    Make_uniform.cpp
    normal.cpp
    uniform.cpp
)

add_subdirectory(random_internal_cpu)
if(USE_CUDA)
  add_subdirectory(random_internal_gpu)
endif()
