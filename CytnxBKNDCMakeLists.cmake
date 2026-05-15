

######################################################################
### Find BLAS and LAPACK
######################################################################
if( NOT (DEFINED BLAS_LIBRARIES AND DEFINED LAPACK_LIBRARIES AND DEFINED LAPACKE_LIBRARIES))
  if (USE_MKL)
    #message(STATUS "ENV{MKLROOT}: $ENV{MKLROOT}")
    # Set MKL interface to LP64 by default, but allow ILP64
    set(MKL_INTERFACE "lp64" CACHE STRING "MKL interface (lp64 or ilp64)")
    set(CYTNX_VARIANT_INFO "${CYTNX_VARIANT_INFO} UNI_MKL")

    set(MKL_ROOT $ENV{MKLROOT})
    message(STATUS "MKL_ROOT: ${MKL_ROOT}")
    message(STATUS "MKL_INTERFACE: ${MKL_INTERFACE}")
    if(MKL_INTERFACE STREQUAL "ilp64")
        set(BLA_VENDOR Intel10_64ilp)
    else()
        set(BLA_VENDOR Intel10_64lp)
    endif()
    message(STATUS "BLA_VENDOR: ${BLA_VENDOR}")
    find_package( BLAS REQUIRED)
    find_package( LAPACK REQUIRED)

    message( STATUS "LAPACK found: ${LAPACK_LIBRARIES}")

    #find_package(MKL CONFIG REQUIRED)
    #Provides available list of targets based on input
    #message(STATUS "MKL_IMPORTED_TARGETS: ${MKL_IMPORTED_TARGETS}")
  #  target_compile_options(cytnx PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
  #  target_include_directories(cytnx PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
  #  target_link_libraries(cytnx PUBLIC  $<LINK_ONLY:MKL::MKL>)
  #  message( STATUS "MKL_LIBRARIES: ${MKL_LIBRARIES}" )
    target_link_libraries(cytnx PUBLIC ${LAPACK_LIBRARIES})
    target_compile_definitions(cytnx PUBLIC UNI_MKL)


  else()
    set(BLA_VENDOR OpenBLAS)
    set(CYTNX_VARIANT_INFO "${CYTNX_VARIANT_INFO} UNI_OPENBLAS")
    message(STATUS "BLA_VENDOR: ${BLA_VENDOR}")
    find_package( BLAS REQUIRED)
    find_package( LAPACK REQUIRED)
    find_package( LAPACKE REQUIRED)
    target_link_libraries(cytnx PUBLIC ${LAPACK_LIBRARIES} ${LAPACKE_LIBRARIES})
    target_include_directories(cytnx PUBLIC ${LAPACKE_INCLUDE_DIRS})
    message( STATUS "LAPACK found: ${LAPACK_LIBRARIES}" )
    message( STATUS "LAPACKE Header found: ${LAPACKE_INCLUDE_DIRS}" )
    message( STATUS "LAPACKE Library found: ${LAPACKE_LIBRARIES}" )
  endif()

else()
  set(LAPACK_LIBRARIES  ${BLAS_LIBRARIES}  ${LAPACK_LIBRARIES} ${LAPACKE_LIBRARIES})
  message( STATUS "LAPACK found: ${LAPACK_LIBRARIES}")
  target_link_libraries(cytnx PUBLIC ${LAPACK_LIBRARIES} ${LAPACKE_LIBRARIES})
  message( STATUS "LAPACKE Header found: ${LAPACKE_INCLUDE_DIRS}" )
  message( STATUS "LAPACKE Library found: ${LAPACKE_LIBRARIES}" )
endif()



if (USE_HPTT)
    set(HPTT_SUBMODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/hptt")
    if(NOT EXISTS "${HPTT_SUBMODULE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "thirdparty/hptt submodule missing. "
            "Run: git submodule update --init --recursive")
    endif()

    # Forward cytnx's HPTT_ENABLE_* options to the names hptt's CMakeLists
    # expects (ENABLE_ARM / ENABLE_AVX / ENABLE_IBM / FINE_TUNE).
    set(ENABLE_ARM "${HPTT_ENABLE_ARM}")
    set(ENABLE_AVX "${HPTT_ENABLE_AVX}")
    set(ENABLE_IBM "${HPTT_ENABLE_IBM}")
    set(FINE_TUNE "${HPTT_ENABLE_FINE_TUNE}")

    # Build hptt in-tree. add_subdirectory works with any CMake generator
    # (no BUILD_BYPRODUCTS gymnastics like ExternalProject_Add needs under
    # Ninja), is faster (no separate sub-build), and exposes hptt_static
    # as a real CMake target — so we can drop the manual include_directories,
    # add_dependencies, and absolute-path library link below.
    add_subdirectory("${HPTT_SUBMODULE_DIR}" "${CMAKE_BINARY_DIR}/hptt")

    set(CYTNX_VARIANT_INFO "${CYTNX_VARIANT_INFO} UNI_HPTT")
    message(STATUS " Build HPTT Support: YES")
    message(STATUS " --HPTT option FINE_TUNE: ${HPTT_ENABLE_FINE_TUNE}")
    message(STATUS " --HPTT option ARM: ${HPTT_ENABLE_ARM}")
    message(STATUS " --HPTT option AVX: ${HPTT_ENABLE_AVX}")
    message(STATUS " --HPTT option IBM: ${HPTT_ENABLE_IBM}")
    FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-DUNI_HPTT\n" "")
endif() #use_HPTT



#####################################################################
### Dependency of CUTT
#####################################################################
if(USE_CUDA)

    set(CYTNX_VARIANT_INFO "${CYTNX_VARIANT_INFO} UNI_CUDA")

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

    set_target_properties(cytnx PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
                                    )
    set_target_properties(cytnx PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe=--display_error_number -lineinfo -m64")
    #set(CMAKE_CUDA_FLAGS "-Xcompiler=-Wall -Xcompiler=-Wno-deprecated-gpu-targets -Xcudafe=--display_error_number")
    ##set(CMAKE_CUDA_FLAGS "-Xcompiler=-Wall -Wno-deprecated-gpu-targets -Xcudafe=--display_error_number")
    ##  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-DUNI_GPU")
    #  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-arch=sm_50 \
    #      -gencode=arch=compute_50,code=sm_50 \
    #      -gencode=arch=compute_52,code=sm_52 \
    #      -gencode=arch=compute_60,code=sm_60 \
    #      -gencode=arch=compute_61,code=sm_61 \
    #      -gencode=arch=compute_70,code=sm_70 \
    #      -gencode=arch=compute_75,code=sm_75 \
    #      -gencode=arch=compute_75,code=compute_75 ")
    target_compile_definitions(cytnx PUBLIC UNI_GPU)
    target_include_directories(cytnx PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    # CUDA 12+/13 may place Thrust/CUB headers under include/cccl.
    set(_cytnx_cccl_candidates)
    if(DEFINED CUDAToolkit_TARGET_DIR AND NOT "${CUDAToolkit_TARGET_DIR}" STREQUAL "")
      list(APPEND _cytnx_cccl_candidates "${CUDAToolkit_TARGET_DIR}/include/cccl")
    endif()
    foreach(_cuda_inc IN LISTS CUDAToolkit_INCLUDE_DIRS)
      list(APPEND _cytnx_cccl_candidates
        "${_cuda_inc}/cccl"
        "${_cuda_inc}/../include/cccl"
        "${_cuda_inc}/../../include/cccl"
        "${_cuda_inc}/../../../include/cccl")
    endforeach()
    list(REMOVE_DUPLICATES _cytnx_cccl_candidates)

    set(_cytnx_cccl_dir "")
    foreach(_cccl_candidate IN LISTS _cytnx_cccl_candidates)
      get_filename_component(_cccl_candidate_abs "${_cccl_candidate}" ABSOLUTE)
      if(EXISTS "${_cccl_candidate_abs}")
        set(_cytnx_cccl_dir "${_cccl_candidate_abs}")
        break()
      endif()
    endforeach()
    if(NOT "${_cytnx_cccl_dir}" STREQUAL "")
      target_include_directories(cytnx PRIVATE "${_cytnx_cccl_dir}")
      message(STATUS "Detected CCCL headers at: ${_cytnx_cccl_dir}")
    endif()

    target_link_libraries(cytnx PUBLIC CUDA::toolkit)
    target_link_libraries(cytnx PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::curand CUDA::cusolver)
    target_link_libraries(cytnx PUBLIC -lcudadevrt)

    if(USE_CUTENSOR)
        find_package(CUTENSOR REQUIRED)
        if(CUTENSOR_FOUND)
            target_compile_definitions(cytnx PUBLIC UNI_CUTENSOR)
            target_include_directories(cytnx PUBLIC ${CUTENSOR_INCLUDE_DIRS})
            target_link_libraries(cytnx PUBLIC ${CUTENSOR_LIBRARIES})
        else()
            message(FATAL_ERROR "cannot find cutensor! please specify cutensor root with -DCUTENSOR_ROOT")
        endif()

        message( STATUS "Build with CuTensor: YES")
        FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-DUNI_CUTENSOR\n" "")
        FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-I${CUTENSOR_INCLUDE_DIRS}\n" "")
        FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "${CUTENSOR_LIBRARIES} -ldl\n" "") # use > to indicate special rt processing
        message( STATUS "CuTensor: libdir:${CUTENSOR_LIBRARY_DIRS} incdir:${CUTENSOR_INCLUDE_DIRS} libs:${CUTENSOR_LIBRARIES}")

    endif()

    if(USE_CUQUANTUM)
        find_package(CUQUANTUM REQUIRED)
        if(CUQUANTUM_FOUND)
            target_compile_definitions(cytnx PUBLIC UNI_CUQUANTUM)
            target_include_directories(cytnx PUBLIC ${CUQUANTUM_INCLUDE_DIRS})
            target_link_libraries(cytnx PUBLIC ${CUQUANTUM_LIBRARIES})
        else()
            message(FATAL_ERROR "cannot find cuquantum! please specify cuquantum root with -DCUQUANTUM_ROOT")
        endif()

        message( STATUS "Build with CuQuantum: YES")
        FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-DUNI_CUQUANTUM\n" "")
        FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-I${CUQUANTUM_INCLUDE_DIRS}\n" "")
        FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "${CUQUANTUM_LIBRARIES} -ldl\n" "") # use > to indicate special rt processing
        message( STATUS "CuQuantum: libdir:${CUQUANTUM_LIBRARY_DIRS} incdir:${CUQUANTUM_INCLUDE_DIRS} libs:${CUQUANTUM_LIBRARIES}")

    endif()





    message( STATUS " Build CUDA Support: YES")
    message( STATUS "  - CUDA Version: ${CUDA_VERSION_STRING}")
    message( STATUS "  - CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")
    message( STATUS "  - Internal macro switch: GPU/CUDA")
    FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-DUNI_GPU\n" "")
    message( STATUS "  - Cudatoolkit include dir: ${CUDAToolkit_INCLUDE_DIRS}")
    FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-I${CUDAToolkit_INCLUDE_DIRS}\n" "")
    message( STATUS "  - Cudatoolkit lib dir: ${CUDAToolkit_LIBRARY_DIR}")
    FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "-L${CUDAToolkit_LIBRARY_DIR}\n" "")
    FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "-Wl,-rpath,${CUDAToolkit_LIBRARY_DIR}\n" "")
    message( STATUS "  - CuSolver library: ${CUDA_cusolver_LIBRARY}")
    FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "${CUDA_cusolver_LIBRARY}\n" "")
    message( STATUS "  - Curand library: ${CUDA_curand_LIBRARY}")
    FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "${CUDA_curand_LIBRARY}\n" "")
    message( STATUS "  - CuBlas library: ${CUDA_cublas_LIBRARY}")
    FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "${CUDA_cublas_LIBRARY}\n" "")
    message( STATUS "  - Cuda rt library: ${CUDA_cudart_static_LIBRARY} -ldl")
    FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "${CUDA_cudart_static_LIBRARY} -ldl\n" "") # use > to indicate special rt processing
    message( STATUS "  - Cuda devrt library: ${CUDA_cudadevrt_LIBRARY} -lrt -lcudadevrt")
    FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "${CUDA_cudadevrt_LIBRARY} -lrt -lcudadevrt\n" "") # use > to indicate special rt processing
    message( STATUS "  - Cuda cusparse library: ${CUDA_cusparse_LIBRARY}")
    FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "${CUDA_cusparse_LIBRARY}\n" "")


else()
    message( STATUS " Build CUDA Support: NO")
endif()

#####################################################################
### Dependency of HPTT
#####################################################################
if(USE_HPTT)
    target_compile_definitions(cytnx PRIVATE UNI_HPTT)

    # hptt's CMakeLists keeps its optimization flags PRIVATE and exposes
    # `hptt::hptt_static` as a namespaced ALIAS, so a plain PRIVATE link
    # is enough — cytnx's compilation picks up the propagated hptt include
    # dir (which is properly wrapped in $<BUILD_INTERFACE:>/$<INSTALL_INTERFACE:>
    # on hptt's side), and hptt's own install(EXPORT hpttTargets) records
    # hptt_static for `install(EXPORT cytnx_targets)`'s referenced-target
    # check. No `$<LINK_ONLY:>` wrap, manual include directory, or extra
    # EXPORT install needed any more.
    target_link_libraries(cytnx PRIVATE hptt::hptt_static)

    # OpenMP stays explicit PUBLIC: libcytnx is a STATIC archive
    # (`add_library(cytnx STATIC)` in CMakeLists.txt) and libhptt.a uses
    # OpenMP internally, so libhptt's OpenMP symbol references (`__kmpc_*`,
    # `omp_*`) stay unresolved until the consumer's final executable link
    # — the consumer's link line therefore needs OpenMP. See commit
    # 5733a441 ("Propagate `-fopenmp` linking flag when using HPTT").
    find_package(OpenMP REQUIRED)
    target_link_libraries(cytnx PUBLIC OpenMP::OpenMP_CXX)
endif()
