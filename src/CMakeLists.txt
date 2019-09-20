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
#  @author Ying-Jer Kao



######################################################################
### ADD SUB-DIRECTORIES
######################################################################

target_sources_local(cytnx
  Accessor.cpp
  Bond.cpp
  BoolStorage.cpp
  ComplexDoubleStorage.cpp
  ComplexFloatStorage.cpp
  DenseUniTensor.cpp
  Device.cpp
  DoubleStorage.cpp
  FermionNetwork.cpp
  FloatStorage.cpp
  Generator.cpp
  Int16Storage.cpp
  Int32Storage.cpp
  Int64Storage.cpp
  Network.cpp
  Network_base.cpp
  RegularNetwork.cpp
  SparseUniTensor.cpp
  Storage.cpp
  Storage_base.cpp
  Symmetry.cpp
  Tensor.cpp
  Type.cpp
  Uint16Storage.cpp
  Uint32Storage.cpp
  Uint64Storage.cpp
  UniTensor.cpp
  UniTensor_base.cpp
  contraction_tree.cpp
  linalg/Add.cpp
  linalg/Cpr.cpp
  linalg/Eigh.cpp
  linalg/Inv.cpp
  linalg/Mul.cpp
  linalg/Svd.cpp
  linalg/linalg_internal_interface.cpp
  linalg/Conj.cpp
  linalg/Diag.cpp
  linalg/Exp.cpp
  linalg/Inv_.cpp
  linalg/Otimes.cpp
  linalg/Svd_truncate.cpp
  linalg/Conj_.cpp
  linalg/Div.cpp
  linalg/Exp_.cpp
  linalg/Matmul.cpp
  linalg/Sub.cpp
  linalg/Tensordot.cpp
)
linalg/linalg_internal_cpu/Add_internal.cpp
linalg/linalg_internal_cpu/Arithmetic_internal.cpp
linalg/linalg_internal_cpu/Conj_inplace_internal.cpp
linalg/linalg_internal_cpu/Cpr_internal.cpp
linalg/linalg_internal_cpu/Diag_internal.cpp
linalg/linalg_internal_cpu/Div_internal.cpp
linalg/linalg_internal_cpu/Eigh_internal.cpp
linalg/linalg_internal_cpu/Exp_internal.cpp
linalg/linalg_internal_cpu/Inv_inplace_internal.cpp
linalg/linalg_internal_cpu/Matmul_internal.cpp
linalg/linalg_internal_cpu/Mul_internal.cpp
linalg/linalg_internal_cpu/Outer_internal.cpp
linalg/linalg_internal_cpu/Sub_internal.cpp
linalg/linalg_internal_cpu/Svd_internal.cpp
../include/Accessor.hpp
../include/Bond.hpp
../include/Device.hpp
../include/Generator.hpp
../include/Network.hpp
../include/Storage.hpp
../include/Symmetry.hpp
../include/Tensor.hpp
../include/Type.hpp
../include/UniTensor.hpp
../include/contraction_tree.hpp
../include/cytnx.hpp
../include/cytnx_error.hpp
../include/intrusive_ptr_base.hpp
../include/linalg/linalg.hpp
../include/linalg/linalg_internal_interface.hpp
../include/utils/cartesian.hpp
../include/utils/complex_arithmetic.hpp
../include/utils/cucomplex_arithmetic.hpp
../include/utils/is.hpp
../include/utils/str_utils.hpp
../include/utils/utils.hpp
../include/utils/utils_internal_interface.hpp
../include/utils/vec_clone.hpp
../include/utils/vec_concatenate.hpp
../include/utils/vec_erase.hpp
../include/utils/vec_intersect.hpp
../include/utils/vec_map.hpp
../include/utils/vec_unique.hpp
../include/utils/vec_where.hpp
../include/linalg/linalg_internal_cpu/Add_internal.hpp
../include/linalg/linalg_internal_cpu/Arithmetic_internal.hpp
../include/linalg/linalg_internal_cpu/Conj_inplace_internal.hpp
../include/linalg/linalg_internal_cpu/Cpr_internal.hpp
../include/linalg/linalg_internal_cpu/Diag_internal.hpp
../include/linalg/linalg_internal_cpu/Div_internal.hpp
../include/linalg/linalg_internal_cpu/Eigh_internal.hpp
../include/linalg/linalg_internal_cpu/Exp_internal.hpp
../include/linalg/linalg_internal_cpu/Inv_inplace_internal.hpp
../include/linalg/linalg_internal_cpu/Matmul_internal.hpp
../include/linalg/linalg_internal_cpu/Mul_internal.hpp
../include/linalg/linalg_internal_cpu/Outer_internal.hpp
../include/linalg/linalg_internal_cpu/Sub_internal.hpp
../include/linalg/linalg_internal_cpu/Svd_internal.hpp
../include/linalg/linalg_internal_gpu/cuAdd_internal.hpp
../include/linalg/linalg_internal_gpu/cuArithmetic_internal.hpp
../include/linalg/linalg_internal_gpu/cuConj_inplace_internal.hpp
../include/linalg/linalg_internal_gpu/cuCpr_internal.hpp
../include/linalg/linalg_internal_gpu/cuDiag_internal.hpp
../include/linalg/linalg_internal_gpu/cuDiv_internal.hpp
../include/linalg/linalg_internal_gpu/cuEigh_internal.hpp
../include/linalg/linalg_internal_gpu/cuExp_internal.hpp
../include/linalg/linalg_internal_gpu/cuInv_inplace_internal.hpp
../include/linalg/linalg_internal_gpu/cuMatmul_internal.hpp
../include/linalg/linalg_internal_gpu/cuMul_internal.hpp
../include/linalg/linalg_internal_gpu/cuOuter_internal.hpp
../include/linalg/linalg_internal_gpu/cuSub_internal.hpp
../include/linalg/linalg_internal_gpu/cuSvd_internal.hpp
linalg/linalg_internal_gpu/cuAdd_internal.cu
linalg/linalg_internal_gpu/cuArithmetic_internal.cu
linalg/linalg_internal_gpu/cuConj_inplace_internal.cu
linalg/linalg_internal_gpu/cuCpr_internal.cu
linalg/linalg_internal_gpu/cuDiag_internal.cu
linalg/linalg_internal_gpu/cuDiv_internal.cu
linalg/linalg_internal_gpu/cuEigh_internal.cu
linalg/linalg_internal_gpu/cuExp_internal.cu
linalg/linalg_internal_gpu/cuInv_inplace_internal.cu
linalg/linalg_internal_gpu/cuMatmul_internal.cu
linalg/linalg_internal_gpu/cuMul_internal.cu
linalg/linalg_internal_gpu/cuOuter_internal.cu
linalg/linalg_internal_gpu/cuSub_internal.cu
linalg/linalg_internal_gpu/cuSvd_internal.cu
