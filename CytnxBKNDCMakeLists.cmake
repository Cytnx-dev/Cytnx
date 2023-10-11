
if (USE_MKL)
  option(MKL_SDL "Link to a single MKL dynamic libary." ON)
  option(MKL_MLT "Use multi-threading libary. [Default]" ON)
  mark_as_advanced(MKL_SDL MKL_MLT)
  set(CYTNX_VARIANT_INFO "${CYTNX_VARIANT_INFO} UNI_MKL")
  target_compile_definitions(cytnx PUBLIC UNI_MKL)
  target_compile_definitions(cytnx PUBLIC MKL_ILP64)
endif() #use_mkl

######################################################################
### Find BLAS and LAPACK
######################################################################
if( NOT (DEFINED BLAS_LIBRARIES AND DEFINED LAPACK_LIBRARIES))
  if (USE_MKL)
    #set(BLA_VENDOR Intel10_64ilp)
    set(BLA_VENDOR Intel10_64_dyn)
    find_package( BLAS REQUIRED)
    find_package( LAPACK REQUIRED)
    #find_package(MKL REQUIRED)
    target_link_libraries(cytnx PUBLIC ${LAPACK_LIBRARIES})
    message( STATUS "LAPACK found: ${LAPACK_LIBRARIES}" )
  else()
    set(BLA_VENDOR OpenBLAS)
    find_package( BLAS REQUIRED)
    find_package( LAPACK REQUIRED)
    target_link_libraries(cytnx PUBLIC ${LAPACK_LIBRARIES})
    message( STATUS "LAPACK found: ${LAPACK_LIBRARIES}" )
  endif()

else()
  set(LAPACK_LIBRARIES  ${BLAS_LIBRARIES}  ${LAPACK_LIBRARIES})
  message( STATUS "LAPACK found: ${LAPACK_LIBRARIES}")
  target_link_libraries(cytnx PUBLIC ${LAPACK_LIBRARIES})
endif()



if (USE_HPTT)
    option(HPTT_ENABLE_ARM "HPTT option ARM" OFF)
    option(HPTT_ENABLE_AVX "HPTT option AVX" OFF)
    option(HPTT_ENABLE_IBM "HPTT option IBM" OFF)
    option(HPTT_ENABLE_FINE_TUNE "HPTT option FINE_TUNE" OFF)
    set(CYTNX_VARIANT_INFO "${CYTNX_VARIANT_INFO} UNI_HPTT")
    ExternalProject_Add(hptt
    PREFIX hptt_src
    GIT_REPOSITORY https://github.com/kaihsin/hptt.git
    GIT_TAG fc9c8cb9b71f4f6d16aad435bdce20025b342a73
    BINARY_DIR hptt_src/build
    INSTALL_DIR hptt
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DENABLE_ARM=${HPTT_ENABLE_ARM} -DENABLE_AVX=${HPTT_ENABLE_AVX} -DENABLE_IBM=${HPTT_ENABLE_IBM} -DFINE_TUNE=${HPTT_ENABLE_FINE_TUNE}
    )
    message( STATUS " Build HPTT Support: YES")
    message( STATUS " --HPTT option FINE_TUNE: ${HPTT_ENABLE_FINE_TUNE}")
    message( STATUS " --HPTT option ARM: ${HPTT_ENABLE_ARM}")
    message( STATUS " --HPTT option AVX: ${HPTT_ENABLE_AVX}")
    message( STATUS " --HPTT option IBM: ${HPTT_ENABLE_IBM}")
    FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-DUNI_HPTT\n" "")
endif() #use_HPTT

if (USE_CUDA)
    if (USE_CUTT)
      option(CUTT_ENABLE_FINE_TUNE "CUTT option FINE_TUNE" OFF)
      option(CUTT_ENABLE_NVTOOLS "CUTT option NVTOOLS" OFF)
      option(CUTT_NO_ALIGN_ALLOC "CUTT option NO_ALIGN_ALLIC" OFF)
      set(CYTNX_VARIANT_INFO "${CYTNX_VARIANT_INFO} UNI_CUTT")
      ExternalProject_Add(cutt
        PREFIX cutt_src
        GIT_REPOSITORY https://github.com/kaihsin/cutt.git
        GIT_TAG 27ed59a42f2610923084c4687327d00f4c2d1d2d
        BINARY_DIR cutt_src/build
        INSTALL_DIR cutt
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DCMAKE_BUILD_TYPE=Release -DNO_ALIGN_ALLOC=${CUTT_NO_ALIGN_ALLOC} -DENABLE_NVTOOLS=${CUTT_ENABLE_NVTOOLS} -DFINE_TUNE=${CUTT_ENABLE_FINE_TUNE} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      )
    endif()
endif()


#####################################################################
### Dependency of CUTT
#####################################################################
if(USE_CUDA)

    set(CYTNX_VARIANT_INFO "${CYTNX_VARIANT_INFO} UNI_CUDA")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
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
    set_property(TARGET cytnx PROPERTY CUDA_ARCHITECTURES native)
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


    if(USE_CUTT)
        ExternalProject_Get_Property(cutt install_dir)
        include_directories(${install_dir}/include)
        message(STATUS "cutt install dir: ${install_dir}")
        add_dependencies(cytnx cutt)
        set_property(TARGET cytnx PROPERTY CUDA_ARCHITECTURES 52 53 60 61 62 70 72 75 80 86)
        target_compile_definitions(cytnx PRIVATE UNI_CUTT)
        target_link_libraries(cytnx PUBLIC ${install_dir}/lib/libcutt.a)
        # relocate cutt
        install(DIRECTORY ${CMAKE_BINARY_DIR}/cutt DESTINATION ${CMAKE_INSTALL_PREFIX})

        message( STATUS " Build CUTT Support: YES")
        message( STATUS " --CUTT option FINE_TUNE: ${CUTT_ENABLE_FINE_TUNE}")
        message( STATUS " --CUTT option NVTOOLS: ${CUTT_ENABLE_NVTOOLS}")
        message( STATUS " --CUTT option NO_ALIGN_ALLOC: ${HPTT_NO_ALIGN_ALLOC}")
        FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-DUNI_CUTT\n" "")

    endif()

    if(USE_MAGMA)
        find_package( MAGMA REQUIRED)
        if(NOT MAGMA_FOUND)
            message(FATAL_ERROR "MAGMA not found!")
        endif()
        message(STATUS "^^^magma root aft: ${MAGMA_ROOT}")
        message(STATUS "^^^magma inc dr: ${MAGMA_INCLUDE_DIRS}")
        message(STATUS "^^^magma lib dr: ${MAGMA_LIBRARY_DIRS}")
        message(STATUS "^^^magma libs: ${MAGMA_LIBRARIES}")
        #add_dependencies(cytnx magma)
        target_include_directories(cytnx PRIVATE ${MAGMA_INCLUDE_DIRS})
        target_compile_definitions(cytnx PRIVATE UNI_MAGMA)
        target_link_libraries(cytnx PUBLIC ${MAGMA_LIBRARIES})

        message( STATUS "Build with MAGMA: YES")
        FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-DUNI_MAGMA\n" "")
        FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-I${MAGMA_INCLUDE_DIRS}\n" "")
        FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "${MAGMA_LIBRARIES} -ldl\n" "") # use > to indicate special rt processing
        message( STATUS "MAGMA: libdir:${MAGMA_LIBRARY_DIRS} incdir:${MAGMA_INCLUDE_DIRS} libs:${MAGMA_LIBRARIES}")
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
    include_directories(${install_dir}/include)
    message(STATUS "hptt install dir: ${install_dir}")
    add_dependencies(cytnx hptt)
    target_compile_definitions(cytnx PRIVATE UNI_HPTT)
    target_link_libraries(cytnx PUBLIC ${install_dir}/lib/libhptt.a)

    # relocate hptt
    install(DIRECTORY ${CMAKE_BINARY_DIR}/hptt DESTINATION ${CMAKE_INSTALL_PREFIX})
endif()





# ----------------------------------------
# Find OpenMP
if(USE_OMP)
    set(CYTNX_VARIANT_INFO "${CYTNX_VARIANT_INFO} UNI_OMP")

    # append file for python
    FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-DUNI_OMP\n" "")

    if(USE_MKL)
        # if MKL is used, we don't explicitly link to OpenMP
        # it's already linked
        target_compile_definitions(cytnx PRIVATE UNI_OMP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

        message( STATUS " Build OMP Support: YES")
        FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-DUNI_MKL\n" "")

    else()
        find_package( OpenMP )
        if ( OPENMP_FOUND )
            message( STATUS " Build OMP Support: YES")
            if(NOT TARGET OpenMP::OpenMP_CXX)
                find_package(Threads REQUIRED)
                add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
                set_property(TARGET OpenMP::OpenMP_CXX
                            PROPERTY INTERFACE_COMPILE_OPTIONS "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${OpenMP_CXX_FLAGS}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${OpenMP_CXX_FLAGS}>")
                # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
                set_property(TARGET OpenMP::OpenMP_CXX
                            PROPERTY INTERFACE_LINK_LIBRARIES "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${OpenMP_CXX_FLAGS}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${OpenMP_CXX_FLAGS}>" Threads::Threads)

            else()
                set_property(TARGET OpenMP::OpenMP_CXX
                            PROPERTY INTERFACE_COMPILE_OPTIONS "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${OpenMP_CXX_FLAGS}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${OpenMP_CXX_FLAGS}>")
            endif()
            target_link_libraries(cytnx PUBLIC OpenMP::OpenMP_CXX)
            target_compile_definitions(cytnx PRIVATE UNI_OMP)
        else()
            message( STATUS " Build OMP Support: NO  (Not found)")
        endif()
    endif()

else()
    message( STATUS " Build OMP Support: NO")

    if(USE_MKL)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif()

    if(USE_HPTT)
        find_package( OpenMP )
        if ( OPENMP_FOUND )
          if(NOT TARGET OpenMP::OpenMP_CXX)
            find_package(Threads REQUIRED)
            add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
            set_property(TARGET OpenMP::OpenMP_CXX
                        PROPERTY INTERFACE_COMPILE_OPTIONS "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${OpenMP_CXX_FLAGS}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${OpenMP_CXX_FLAGS}>")
            # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
            set_property(TARGET OpenMP::OpenMP_CXX
                        PROPERTY INTERFACE_LINK_LIBRARIES "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${OpenMP_CXX_FLAGS}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${OpenMP_CXX_FLAGS}>" Threads::Threads)

          else()
            set_property(TARGET OpenMP::OpenMP_CXX
                        PROPERTY INTERFACE_COMPILE_OPTIONS "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${OpenMP_CXX_FLAGS}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${OpenMP_CXX_FLAGS}>")
          endif()
          target_link_libraries(cytnx PUBLIC OpenMP::OpenMP_CXX)
        endif()
    endif()

endif()
