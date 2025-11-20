

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
    find_package( BLAS REQUIRED)
    find_package( LAPACK REQUIRED)
    find_package( LAPACKE REQUIRED)
    target_link_libraries(cytnx PUBLIC ${LAPACK_LIBRARIES} ${LAPACKE_LIBRARIES})
    set(LAPACKE_INCLUDE_DIRS ${LAPACKE_DIR_FOUND}/include)
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
    option(HPTT_ENABLE_ARM "HPTT option ARM" OFF)
    option(HPTT_ENABLE_IBM "HPTT option IBM" OFF)
    option(HPTT_ENABLE_AVX "HPTT option AVX" OFF)
    option(HPTT_ENABLE_FINE_TUNE "HPTT option FINE_TUNE" OFF)

    # Use absolute paths so Ninja can track the artifacts produced by the ExternalProject.
    cmake_path(APPEND CMAKE_CURRENT_BINARY_DIR "hptt" OUTPUT_VARIABLE HPTT_PREFIX)
    set(HPTT_STATIC_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}hptt${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(HPTT_SHARED_LIB_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}hptt${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(HPTT_BYPRODUCTS "${HPTT_PREFIX}/lib/${HPTT_STATIC_LIB_NAME}")
    if(NOT WIN32 AND NOT CMAKE_SHARED_LIBRARY_SUFFIX STREQUAL "")
        list(APPEND HPTT_BYPRODUCTS "${HPTT_PREFIX}/lib/${HPTT_SHARED_LIB_NAME}")
    endif()

    set(CYTNX_VARIANT_INFO "${CYTNX_VARIANT_INFO} UNI_HPTT")
    # TODO: Build HPTT from the submodule in the thirdparty folder.
    ExternalProject_Add(hptt
    PREFIX "${HPTT_PREFIX}"
    GIT_REPOSITORY https://github.com/Cytnx-dev/hptt.git
    GIT_TAG 50bc0b65d2bb4751fc88414681363e1995e41b23
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DENABLE_ARM=${HPTT_ENABLE_ARM} -DENABLE_AVX=${HPTT_ENABLE_AVX} -DENABLE_IBM=${HPTT_ENABLE_IBM} -DFINE_TUNE=${HPTT_ENABLE_FINE_TUNE}
    BUILD_BYPRODUCTS ${HPTT_BYPRODUCTS}
    )
    message( STATUS " Build HPTT Support: YES")
    message( STATUS " --HPTT option FINE_TUNE: ${HPTT_ENABLE_FINE_TUNE}")
    message( STATUS " --HPTT option ARM: ${HPTT_ENABLE_ARM}")
    message( STATUS " --HPTT option AVX: ${HPTT_ENABLE_AVX}")
    message( STATUS " --HPTT option IBM: ${HPTT_ENABLE_IBM}")
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
    ExternalProject_Get_Property(hptt install_dir)
    cmake_path(APPEND install_dir "include" OUTPUT_VARIABLE hptt_include_dir)
    cmake_path(APPEND install_dir "lib" OUTPUT_VARIABLE hptt_lib_dir)
    include_directories("${hptt_include_dir}")
    message(STATUS "hptt install dir: ${install_dir}")
    unset(install_dir)
    add_dependencies(cytnx hptt)
    target_compile_definitions(cytnx PRIVATE UNI_HPTT)
    target_link_libraries(cytnx PUBLIC "${hptt_lib_dir}/libhptt.a")

    # XXX: `cytnx` itself doesn't need this linking flag. Why?
    #target_link_options(cytnx INTERFACE -fopenmp)
    #Find OpenMP for HPTT
    find_package(OpenMP REQUIRED)
    target_link_libraries(cytnx PUBLIC OpenMP::OpenMP_CXX)

    # Install HPTT to input CMAKE_INSTALL_PREFIX.
    cmake_path(APPEND CMAKE_INSTALL_INCLUDEDIR "hptt" OUTPUT_VARIABLE hptt_install_include_dir)
    # Suffix the source folder with / to copy the files and folders in the source folder to the
    # destination folder instead of copying the source folder itself.
    # XXX: Do we have to ship the header files of HPTT? Is shipping the static library only enough?
    install(DIRECTORY "${hptt_include_dir}/" DESTINATION "${hptt_install_include_dir}")
    # CMake doesn't combine external static libraries into our static library, so we have to
    # distribute the external static libraries manually.
    install(DIRECTORY "${hptt_lib_dir}/" TYPE LIB)
endif()
