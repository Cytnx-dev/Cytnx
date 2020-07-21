#
#  @file CMakeLists.txt
#  @license
#    Copyright (c) 2013-2017
#    National Taiwan University
#    National Tsing-Hua University
#
#    This file is part of Cytnx.
#
#    Cytnx is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Uni10 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with Cytnx.  If not, see <http://www.gnu.org/licenses/>.
#  @endlicense
#  @brief Main specification file for CMake
#  @author Ying-Jer Kao, Kai-Hsin Wu



######################################################################
### ADD SUB-DIRECTORIES
######################################################################

target_sources_local(cytnx
  PRIVATE
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
    
)

#if(UNI_GPU)
#    target_sourse_local(cytnx
#        PRIVATE
#            Device.cu
#    )
#endif

add_subdirectory(linalg)
add_subdirectory(utils)
add_subdirectory(random)
