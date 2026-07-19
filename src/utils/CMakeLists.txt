target_sources(cytnx
  PRIVATE

    #utils_internal_interface.hpp
    cartesian.cpp
    is.cpp
    str_utils.cpp
    dynamic_arg_resolver.cpp
    utils.cpp
    #utils_internal_interface.cpp
    vec_concatenate.cpp
    vec_erase.cpp
    vec_intersect.cpp
    vec_map.cpp
    vec_range.cpp
    vec_unique.cpp
    vec_where.cpp
    vec2d_col_sort.cpp
    vec_print.cpp
    vec_io.cpp
    vec_argsort.cpp
    vec_sort.cpp
)


if(USE_CUDA)
  target_sources(cytnx
    PRIVATE
  )

  if(USE_CUQUANTUM)
    target_sources(cytnx
    PRIVATE
    cutensornet.hpp
    cutensornet.cu
  )
  endif()
endif()
