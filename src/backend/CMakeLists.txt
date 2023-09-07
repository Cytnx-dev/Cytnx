

######################################################################
### ADD SUB-DIRECTORIES
######################################################################

target_sources_local(cytnx
  PRIVATE
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
)

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
