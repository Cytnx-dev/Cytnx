target_sources_local(cytnx
  PRIVATE
    # cartesian.hpp
    # complex_arithmetic.hpp
    # cucomplex_arithmetic.hpp
    # is.hpp
    # str_utils.hpp
    # utils.hpp
    # utils_internal_interface.hpp
    # vec_clone.hpp
    # vec_concatenate.hpp
    # vec_erase.hpp
    # vec_intersect.hpp
    # vec_map.hpp
    # vec_unique.hpp
    # vec_where.hpp
    cartesian.cpp
    complex_arithmetic.cpp
    is.cpp
    str_utils.cpp
    utils.cpp
    utils_internal_interface.cpp
    vec_clone.cpp
    vec_concatenate.cpp
    vec_erase.cpp
    vec_intersect.cpp
    vec_map.cpp
    vec_range.cpp
    vec_unique.cpp
    vec_where.cpp
    vec2d_col_sort.cpp
)

add_subdirectory(utils_internal_cpu)
if(USE_CUDA)
  add_subdirectory(utils_internal_gpu)
endif()
