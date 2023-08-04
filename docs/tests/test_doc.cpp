#include "cytnx.hpp"
#include <gtest/gtest.h>

// redirect standard output: if you need to output to another file, you can use these
//     macro to redirect the standart output
#define OUT_REDIRECT_BASE \
  coutbuf = cout.rdbuf(); \
  cout.rdbuf(output_file_o.rdbuf());  // redirect std::cout to file

#define OUT_REDIRECT                                                                 \
  string suite_name = testing::UnitTest::GetInstance()->current_test_info()->name(); \
  \ 
	output_file_o.open(output_dir + suite_name + ".out");                        \
  OUT_REDIRECT_BASE

#define OUT_REDIRECT_FILE(OUT_FILE)                   \
  output_file_o.open(output_dir + OUT_FILE + ".out"); \
  OUT_REDIRECT_BASE

#define OUT_RESET      \
  cout.rdbuf(coutbuf); \
  output_file_o.close();

using namespace cytnx;
using namespace std;
using namespace testing;

namespace DocTest {
  const string output_dir = "../../code/cplusplus/outputs/";

  // output redirect
  streambuf *coutbuf = nullptr;
  ofstream output_file_o;

  // 1. Objects behavior
  // 1.1. Everyting is reference
  TEST(Doc, guide_behavior_assign) {
    OUT_REDIRECT
    #include "guide_behavior_assign.cpp"
    OUT_RESET
  }

  TEST(Doc, guide_behavior_clone) {
    OUT_REDIRECT
    #include "guide_behavior_clone.cpp"
    OUT_RESET
  }

  // 1.2 clone
  // 2. Device
  // 2.1 Number of threads
  TEST(Doc, guide_Device_Ncpus) {
    OUT_REDIRECT
    #include "guide_Device_Ncpus.cpp"
    OUT_RESET
  }

  // 2.2 Number of GPUs
  TEST(Doc, guide_Device_Ngpus) {
    OUT_REDIRECT
    #include "guide_Device_Ngpus.cpp"
    OUT_RESET
  }

  // 3. Tensor
  // 3.1 Creating a Tensor
  // 3.1.1 Initialized Tensor
  TEST(Doc, guide_basic_obj_Tensor_1_create_zeros) {
    #include "guide_basic_obj_Tensor_1_create_zeros.cpp"
  }

  TEST(Doc, guide_basic_obj_Tensor_1_create_diff_ways) {
    #include "guide_basic_obj_Tensor_1_create_diff_ways.cpp"
  }

  // 3.1.2 Random Tensor
  TEST(Doc, guide_basic_obj_Tensor_1_create_rand) {
    #include "guide_basic_obj_Tensor_1_create_rand.cpp"
  }

// 3.1.3 Tensor with different dtype and device
#ifdef UNI_GPU
  TEST(Doc, guide_basic_obj_Tensor_1_create_zeros_cuda) {
    #include "guide_basic_obj_Tensor_1_create_zeros_cuda.cpp"
  }
#endif

  // 3.1.4 Type conversion
  TEST(Doc, guide_basic_obj_Tensor_1_create_astype) {
    OUT_REDIRECT
    #include "guide_basic_obj_Tensor_1_create_astype.cpp"
    OUT_RESET
  }

// 3.1.5 Transfer between devices
#ifdef UNI_GPU
  TEST(Doc, guide_basic_obj_Tensor_1_create_to) {
    OUT_REDIRECT
    #include "guide_basic_obj_Tensor_1_create_to.cpp"
    OUT_RESET
  }
#endif

  // 3.1.6 Tensor from Storage
  TEST(Doc, guide_basic_obj_Tensor_1_create_from_storage) {
    #include "guide_basic_obj_Tensor_1_create_from_storage.cpp"
  }

  // 3.2 Manipulating Tensors
  // 3.2.1 reshape
  TEST(Doc, guide_basic_obj_Tensor_2_manip_reshape) {
    OUT_REDIRECT
    #include "guide_basic_obj_Tensor_2_manip_reshape.cpp"
    OUT_RESET
  }

  TEST(Doc, guide_basic_obj_Tensor_2_manip_reshape_) {
    OUT_REDIRECT
    #include "guide_basic_obj_Tensor_2_manip_reshape_.cpp"
    OUT_RESET
  }

  // 3.2.2 permute
  TEST(Doc, guide_basic_obj_Tensor_2_manip_permute) {
    OUT_REDIRECT
    #include "guide_basic_obj_Tensor_2_manip_permute.cpp"
    OUT_RESET
  }

  TEST(Doc, guide_basic_obj_Tensor_2_manip_contiguous) {
    OUT_REDIRECT
    #include "guide_basic_obj_Tensor_2_manip_contiguous.cpp"
    OUT_RESET
  }

  // 3.3 Accessing elements
  // 3.3.1 Get elements
  TEST(Doc, guide_basic_obj_Tensor_3_access_slice_get) {
    OUT_REDIRECT
    #include "guide_basic_obj_Tensor_3_access_slice_get.cpp"
    OUT_RESET
  }

  TEST(Doc, guide_basic_obj_Tensor_3_access_item) {
    OUT_REDIRECT
    #include "guide_basic_obj_Tensor_3_access_item.cpp"
    OUT_RESET
  }

  // 3.3.2 Get elements
  TEST(Doc, guide_basic_obj_Tensor_3_access_slice_set) {
    OUT_REDIRECT
    #include "guide_basic_obj_Tensor_3_access_slice_set.cpp"
    OUT_RESET
  }

  // 3.3.3 Low-level API (C++ only)
  /*
  TEST(Doc, guide_basic_obj_Tensor_3_access_c_accessor) {
    #include "guide_basic_obj_Tensor_3_access_c_accessor.cpp"
  }

  TEST(Doc, guide_basic_obj_Tensor_3_access_c_operator) {
    #include "guide_basic_obj_Tensor_3_access_c_operator.cpp"
  }

  TEST(Doc, guide_basic_obj_Tensor_3_access_c_get_set) {
  }
    #include "guide_basic_obj_Tensor_3_access_c_get_set.cpp"
  */

  // 3.4 Tensor arithmetic
  // 3.4.2 Tensor-scalar arithmetic
  TEST(Doc, guide_basic_obj_Tensor_4_arithmetic_tensor_scalar) {
    OUT_REDIRECT
    #include "guide_basic_obj_Tensor_4_arithmetic_tensor_scalar.cpp"
    OUT_RESET
  }

  // 3.4.3 Tensor-Tensor arithmetic
  TEST(Doc, guide_basic_obj_Tensor_4_arithmetic_tensor_tensor) {
    OUT_REDIRECT
    #include "guide_basic_obj_Tensor_4_arithmetic_tensor_tensor.cpp"
    OUT_RESET
  }

  // 3.4.4 Equivalent APIs (C++ only)
  TEST(Doc, guide_basic_obj_Tensor_4_arithmetic_Add) {
    #include "guide_basic_obj_Tensor_4_arithmetic_Add.cpp"
  }

  // 3.6 Appending elements
  TEST(Doc, guide_basic_obj_Tensor_6_app_scalar) {
    OUT_REDIRECT
    #include "guide_basic_obj_Tensor_6_app_scalar.cpp"
    OUT_RESET
  }

  TEST(Doc, guide_basic_obj_Tensor_6_app_tensor) {
    OUT_REDIRECT
    #include "guide_basic_obj_Tensor_6_app_tensor.cpp"
    OUT_RESET
  }

  // 3.7 Save/Load
  // 3.7.1 Save a Tensor
  TEST(Doc, guide_basic_obj_Tensor_7_io_Save) {
    OUT_REDIRECT
    #include "guide_basic_obj_Tensor_7_io_Save.cpp"
    OUT_RESET
  }

  // 3.7.2 Load a Tensor
  TEST(Doc, guide_basic_obj_Tensor_7_io_Load) {
    OUT_REDIRECT
    #include "guide_basic_obj_Tensor_7_io_Load.cpp"
    OUT_RESET
  }

  // 3.8 When will data be copied?
  // 3.8.1 Reference to & Copy of objects
  TEST(Doc, guide_basic_obj_Tensor_8_cp_assign) {
    OUT_REDIRECT
    #include "guide_basic_obj_Tensor_8_cp_assign.cpp"
    OUT_RESET
  }

  TEST(Doc, guide_basic_obj_Tensor_8_cp_clone) {
    OUT_REDIRECT
    #include "guide_basic_obj_Tensor_8_cp_clone.cpp"
    OUT_RESET
  }

  // 3.8.2 Permute
  TEST(Doc, guide_basic_obj_Tensor_8_cp_permute) {
    OUT_REDIRECT_FILE("guide_basic_obj_Tensor_8_cp_permute-1")
    #include "guide_basic_obj_Tensor_8_cp_permute-1.cpp"
    OUT_RESET OUT_REDIRECT_FILE("guide_basic_obj_Tensor_8_cp_permute-2")
    #include "guide_basic_obj_Tensor_8_cp_permute-2.cpp"
    OUT_RESET OUT_REDIRECT_FILE("guide_basic_obj_Tensor_8_cp_permute-3")
    #include "guide_basic_obj_Tensor_8_cp_permute-3.cpp"
    OUT_RESET
  }

  // 3.8.3 Contiguous
  TEST(Doc, guide_basic_obj_Tensor_8_cp_contiguous){
    OUT_REDIRECT_FILE("guide_basic_obj_Tensor_8_cp_contiguous-1")
    #include "guide_basic_obj_Tensor_8_cp_contiguous-1.cpp"
    OUT_RESET OUT_REDIRECT_FILE("guide_basic_obj_Tensor_8_cp_contiguous-2")
    #include "guide_basic_obj_Tensor_8_cp_contiguous-2.cpp"
    OUT_RESET
  }

  // 4. Storage
  // 4.1 Creating a Storage
  TEST(Doc, guide_basic_obj_Storage_1_create_create) {
    OUT_REDIRECT
    #include "guide_basic_obj_Storage_1_create_create.cpp"
    OUT_RESET
  }

  // 4.1.1 Type conversion
  TEST(Doc, guide_basic_obj_Storage_1_create_astype) {
    OUT_REDIRECT
    #include "guide_basic_obj_Storage_1_create_astype.cpp"
    OUT_RESET
  }

// 4.1.2 Transfer between devices
#ifdef UNI_GPU
  TEST(Doc, guide_basic_obj_Storage_1_create_to) {
    OUT_REDIRECT
    #include "guide_basic_obj_Storage_1_create_to.cpp"
    OUT_RESET
  }
#endif

  // 4.1.3 Get Storage of Tensor
  TEST(Doc, guide_basic_obj_Storage_1_create_get_storage) {
    OUT_REDIRECT
    #include "guide_basic_obj_Storage_1_create_get_storage.cpp"
    OUT_RESET
  }

  // 4.2 Accessing elements
  // 4.2.1 Get/Set elements
  TEST(Doc, guide_basic_obj_Storage_2_access_access) {
    OUT_REDIRECT
    #include "guide_basic_obj_Storage_2_access_access.cpp"
    OUT_RESET
  }

  TEST(Doc, guide_basic_obj_Storage_2_access_at) {
    OUT_REDIRECT
    #include "guide_basic_obj_Storage_2_access_at.cpp"
    OUT_RESET
  }

  // 4.2.2 Get raw-pointer (C++ only)
  TEST(Doc, guide_basic_obj_Storage_2_access_ptr_T) {
    #include "guide_basic_obj_Storage_2_access_ptr_T.cpp"
  }

  TEST(Doc, guide_basic_obj_Storage_2_access_ptr_void) {
    #include "guide_basic_obj_Storage_2_access_ptr_void.cpp"
  }

  // 4.3 Increase size
  // 4.3.1 append
  TEST(Doc, guide_basic_obj_Storage_3_expand_append) {
    OUT_REDIRECT
    #include "guide_basic_obj_Storage_3_expand_append.cpp"
    OUT_RESET
  }

  // 4.3.2 resize
  TEST(Doc, guide_basic_obj_Storage_3_expand_resize) {
    OUT_REDIRECT
    #include "guide_basic_obj_Storage_3_expand_resize.cpp"
    OUT_RESET
  }

// 4.4 From/To C++ .vector
#ifdef UNI_GPU
  TEST(Doc, guide_basic_obj_Storage_4_vec_from_vec) {
    OUT_REDIRECT
    #include "guide_basic_obj_Storage_4_vec_from_vec.cpp"
    OUT_RESET
  }
#endif

  TEST(Doc, guide_basic_obj_Storage_4_vec_to_vec) {
    OUT_REDIRECT
    #include "guide_basic_obj_Storage_4_vec_to_vec.cpp"
    OUT_RESET
  }

  // 4.5 Save/Load
  // 4.5.1 Save a Storage
  TEST(Doc, guide_basic_obj_Storage_5_io_Save) {
    #include "guide_basic_obj_Storage_5_io_Save.cpp"
  }

  // 4.5.2 Load a Storage
  TEST(Doc, guide_basic_obj_Storage_5_io_Load) {
    OUT_REDIRECT
    #include "guide_basic_obj_Storage_5_io_Load.cpp"
    OUT_RESET
  }

  // 4.5.3 Save & load from/to binary
  TEST(Doc, guide_basic_obj_Storage_5_io_from_to_file) {
    OUT_REDIRECT
    #include "guide_basic_obj_Storage_5_io_from_to_file.cpp"
    OUT_RESET
  }

  // 5. Scalar
  // 5.1 Define/Declare a Scalar
  TEST(Doc, guide_basic_obj_Scalar_from_c_var) {
    OUT_REDIRECT
    #include "guide_basic_obj_Scalar_from_c_var.cpp"
    OUT_RESET
  }

  TEST(Doc, guide_basic_obj_Scalar_create) {
    OUT_REDIRECT
    #include "guide_basic_obj_Scalar_create.cpp"
    OUT_RESET
  }

  TEST(Doc, guide_basic_obj_Scalar_to_c_var) {
    OUT_REDIRECT
    #include "guide_basic_obj_Scalar_to_c_var.cpp"
    OUT_RESET
  }

  // 5.2 Change date type
  TEST(Doc, guide_basic_obj_Scalar_astype) {
    OUT_REDIRECT
    #include "guide_basic_obj_Scalar_astype.cpp"
    OUT_RESET
  }

  // 5.3 Application scenarios
  TEST(Doc, guide_basic_obj_Scalar_application) {
    OUT_REDIRECT
    #include "guide_basic_obj_Scalar_application.cpp"
    OUT_RESET
  }

  // 7. UniTensor
  // 7.4 Bond
  // 7.4.1 Symmetry object
  TEST(Doc, guide_uniten_bond_symobj) {
    OUT_REDIRECT
    #include "guide_uniten_bond_symobj.cpp"
    OUT_RESET
  }

  // 7.4.2 Creating Bonds with quantum numbers
  TEST(Doc, guide_uniten_bond_sym_bond) {
    OUT_REDIRECT
    #include "guide_uniten_bond_sym_bond.cpp"
    OUT_RESET
  }

  // 7.8 Get/set UniTensor element
  // 7.8.1 UniTensor without symmetries
  TEST(Doc, guide_uniten_elements_at_get) {
    OUT_REDIRECT
    #include "guide_uniten_elements_at_get.cpp"
    OUT_RESET
  }

  TEST(Doc, guide_uniten_elements_at_set) {
    OUT_REDIRECT
    #include "guide_uniten_elements_at_set.cpp"
    OUT_RESET
  }

  // 7.8.1 UniTensor with symmetries
  TEST(Doc, user_guide_7_8_2_ex1) {
    // need to add
  }

  // 8. Contraction
  // 8.1 Network
  // 8.1.2 Put UniTensors and Launch
  TEST(Doc, guide_contraction_network_launch) {
    OUT_REDIRECT_FILE("guide_contraction_network_PutUniTensor")
    #include "guide_contraction_network_PutUniTensor.cpp"
    OUT_RESET
    #include "guide_contraction_network_launch.cpp"
  }

  // 10. Iterative solver
  // 10.1 LinOp class
  TEST(Doc, guide_itersol_LinOp_Dot) {
    OUT_REDIRECT
    #include "guide_itersol_LinOp_Dot.cpp"
    OUT_RESET
  }

  // 10.1.2 Inherit the LinOp class
  TEST(Doc, guide_itersol_LinOp_matvec) {
    OUT_REDIRECT
    #include "guide_itersol_LinOp_matvec.cpp"
    OUT_RESET
  }

  // 10.2 Lanczos solver
  TEST(Doc, guide_itersol_Lanczos_Lanczos) {
    OUT_REDIRECT
    #include "guide_itersol_Lanczos_Lanczos.cpp"
    OUT_RESET
  }

}  // namespace DocTest
