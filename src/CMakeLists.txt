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
)

