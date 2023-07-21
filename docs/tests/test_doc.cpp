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
  TEST(Doc, user_guide_1_1_ex1) {
    OUT_REDIRECT
    #include "user_guide_1_1_ex1.cpp"
    OUT_RESET
  }

  TEST(Doc, user_guide_1_2_ex1) {
    OUT_REDIRECT
    #include "user_guide_1_2_ex1.cpp"
    OUT_RESET
  }

  // 1.2 clone
  // 2. Device
  // 2.1 Number of threads
  TEST(Doc, user_guide_2_1_ex1) {
    OUT_REDIRECT
    #include "user_guide_2_1_ex1.cpp"
    OUT_RESET
  }

  // 2.2 Number of GPUs
  TEST(Doc, user_guide_2_2_ex1) {
    OUT_REDIRECT
    #include "user_guide_2_2_ex1.cpp"
    OUT_RESET
  }

  // 3. Tensor
  // 3.1 Creating a Tensor
  // 3.1.1 Initialized Tensor
  TEST(Doc, user_guide_3_1_1_ex1) {
    #include "user_guide_3_1_1_ex1.cpp"
  }

  TEST(Doc, user_guide_3_1_1_ex2) {
    #include "user_guide_3_1_1_ex2.cpp"
  }

  // 3.1.2 Random Tensor
  TEST(Doc, user_guide_3_1_2_ex1) {
    #include "user_guide_3_1_2_ex1.cpp"
  }

// 3.1.3 Tensor with different dtype and device
#ifdef UNI_GPU
  TEST(Doc, user_guide_3_1_3_ex1) {
    #include "user_guide_3_1_3_ex1.cpp"
  }
#endif

  // 3.1.4 Type conversion
  TEST(Doc, user_guide_3_1_4_ex1) {
    OUT_REDIRECT
    #include "user_guide_3_1_4_ex1.cpp"
    OUT_RESET
  }

// 3.1.5 Transfer between devices
#ifdef UNI_GPU
  TEST(Doc, user_guide_3_1_5_ex1) {
    OUT_REDIRECT
    #include "user_guide_3_1_5_ex1.cpp"
    OUT_RESET
  }
#endif

  // 3.1.6 Tensor from Storage
  TEST(Doc, user_guide_3_1_6_ex1) {
    #include "user_guide_3_1_6_ex1.cpp"
  }

  // 3.2 Manipulating Tensors
  // 3.2.1 reshape
  TEST(Doc, user_guide_3_2_1_ex1) {
    OUT_REDIRECT
    #include "user_guide_3_2_1_ex1.cpp"
    OUT_RESET
  }

  TEST(Doc, user_guide_3_2_1_ex2) {
    OUT_REDIRECT
    #include "user_guide_3_2_1_ex2.cpp"
    OUT_RESET
  }

  // 3.2.2 permute
  TEST(Doc, user_guide_3_2_2_ex1) {
    OUT_REDIRECT
    #include "user_guide_3_2_2_ex1.cpp"
    OUT_RESET
  }

  TEST(Doc, user_guide_3_2_2_ex2) {
    OUT_REDIRECT
    #include "user_guide_3_2_2_ex2.cpp"
    OUT_RESET
  }

  // 3.3 Accessing elements
  // 3.3.1 Get elements
  TEST(Doc, user_guide_3_3_1_ex1) {
    OUT_REDIRECT
    #include "user_guide_3_3_1_ex1.cpp"
    OUT_RESET
  }

  TEST(Doc, user_guide_3_3_1_ex2) {
    OUT_REDIRECT
    #include "user_guide_3_3_1_ex2.cpp"
    OUT_RESET
  }

  // 3.3.2 Get elements
  TEST(Doc, user_guide_3_3_2_ex1) {
    OUT_REDIRECT
    #include "user_guide_3_3_2_ex1.cpp"
    OUT_RESET
  }

  // 3.3.3 Low-level API (C++ only)
  /*
  TEST(Doc, user_guide_3_3_3_ex1) {
    #include "user_guide_3_3_3_ex1.cpp"
  }

  TEST(Doc, user_guide_3_3_3_ex2) {
    #include "user_guide_3_3_3_ex2.cpp"
  }

  TEST(Doc, user_guide_3_3_3_ex3) {
  }
    #include "user_guide_3_3_3_ex3.cpp"
  */

  // 3.4 Tensor arithmetic
  // 3.4.2 Tensor-scalar arithmetic
  TEST(Doc, user_guide_3_4_2_ex1) {
    OUT_REDIRECT
    #include "user_guide_3_4_2_ex1.cpp"
    OUT_RESET
  }

  // 3.4.3 Tensor-Tensor arithmetic
  TEST(Doc, user_guide_3_4_3_ex1) {
    OUT_REDIRECT
    #include "user_guide_3_4_3_ex1.cpp"
    OUT_RESET
  }

  // 3.4.4 Equivalent APIs (C++ only)
  TEST(Doc, user_guide_3_4_4_ex1) {
    #include "user_guide_3_4_4_ex1.cpp"
  }

  // 3.6 Appending elements
  TEST(Doc, user_guide_3_6_ex1) {
    OUT_REDIRECT
    #include "user_guide_3_6_ex1.cpp"
    OUT_RESET
  }

  TEST(Doc, user_guide_3_6_ex2) {
    OUT_REDIRECT
    #include "user_guide_3_6_ex2.cpp"
    OUT_RESET
  }

  // 3.7 Save/Load
  // 3.7.1 Save a Tensor
  TEST(Doc, user_guide_3_7_1_ex1) {
    OUT_REDIRECT
    #include "user_guide_3_7_1_ex1.cpp"
    OUT_RESET
  }

  // 3.7.2 Load a Tensor
  TEST(Doc, user_guide_3_7_2_ex1) {
    OUT_REDIRECT
    #include "user_guide_3_7_2_ex1.cpp"
    OUT_RESET
  }

  // 3.8 When will data be copied?
  // 3.8.1 Reference to & Copy of objects
  TEST(Doc, user_guide_3_8_1_ex1) {
    OUT_REDIRECT
    #include "user_guide_3_8_1_ex1.cpp"
    OUT_RESET
  }

  TEST(Doc, user_guide_3_8_1_ex2) {
    OUT_REDIRECT
    #include "user_guide_3_8_1_ex2.cpp"
    OUT_RESET
  }

  // 3.8.2 Permute
  TEST(Doc, user_guide_3_8_2_ex1_2_3) {
    OUT_REDIRECT_FILE("user_guide_3_8_2_ex1")
    #include "user_guide_3_8_2_ex1.cpp"
    OUT_RESET OUT_REDIRECT_FILE("user_guide_3_8_2_ex2")
    #include "user_guide_3_8_2_ex2.cpp"
    OUT_RESET OUT_REDIRECT_FILE("user_guide_3_8_2_ex3")
    #include "user_guide_3_8_2_ex3.cpp"
    OUT_RESET
  }

  // 3.8.2 Contiguous
  TEST(Doc, user_guide_3_8_3_ex1_2){
    OUT_REDIRECT_FILE("user_guide_3_8_3_ex1")
    #include "user_guide_3_8_3_ex1.cpp"
    OUT_RESET OUT_REDIRECT_FILE("user_guide_3_8_3_ex2")
    #include "user_guide_3_8_3_ex2.cpp"
    OUT_RESET
  }

  // 4. Storage
  // 4.1 Creating a Storage
  TEST(Doc, user_guide_4_1_ex1) {
    OUT_REDIRECT
    #include "user_guide_4_1_ex1.cpp"
    OUT_RESET
  }

  // 4.1.1 Type conversion
  TEST(Doc, user_guide_4_1_1_ex1) {
    OUT_REDIRECT
    #include "user_guide_4_1_1_ex1.cpp"
    OUT_RESET
  }

// 4.1.2 Transfer between devices
#ifdef UNI_GPU
  TEST(Doc, user_guide_4_1_2_ex1) {
    OUT_REDIRECT
    #include "user_guide_4_1_2_ex1.cpp"
    OUT_RESET
  }
#endif

  // 4.1.3 Get Storage of Tensor
  TEST(Doc, user_guide_4_1_3_ex1) {
    OUT_REDIRECT
    #include "user_guide_4_1_3_ex1.cpp"
    OUT_RESET
  }

  // 4.2 Accessing elements
  // 4.2.1 Get/Set elements
  TEST(Doc, user_guide_4_2_1_ex1) {
    OUT_REDIRECT
    #include "user_guide_4_2_1_ex1.cpp"
    OUT_RESET
  }

  TEST(Doc, user_guide_4_2_1_ex2) {
    OUT_REDIRECT
    #include "user_guide_4_2_1_ex2.cpp"
    OUT_RESET
  }

  // 4.2.2 Get raw-pointer (C++ only)
  TEST(Doc, user_guide_4_2_2_ex1) {
    #include "user_guide_4_2_2_ex1.cpp"
  }

  TEST(Doc, user_guide_4_2_2_ex2) {
    #include "user_guide_4_2_2_ex2.cpp"
  }

  // 4.3 Increase size
  // 4.3.1 append
  TEST(Doc, user_guide_4_3_1_ex1) {
    OUT_REDIRECT
    #include "user_guide_4_3_1_ex1.cpp"
    OUT_RESET
  }

  // 4.3.2 resize
  TEST(Doc, user_guide_4_3_2_ex1) {
    OUT_REDIRECT
    #include "user_guide_4_3_2_ex1.cpp"
    OUT_RESET
  }

// 4.4 From/To C++ .vector
#ifdef UNI_GPU
  TEST(Doc, user_guide_4_4_ex1) {
    OUT_REDIRECT
    #include "user_guide_4_4_ex1.cpp"
    OUT_RESET
  }
#endif

  TEST(Doc, user_guide_4_4_ex2) {
    OUT_REDIRECT
    #include "user_guide_4_4_ex2.cpp"
    OUT_RESET
  }

  // 4.5 Save/Load
  // 4.5.1 Save a Storage
  TEST(Doc, user_guide_4_5_1_ex1) {
    #include "user_guide_4_5_1_ex1.cpp"
  }

  // 4.5.2 Load a Storage
  TEST(Doc, user_guide_4_5_2_ex1) {
    OUT_REDIRECT
    #include "user_guide_4_5_2_ex1.cpp"
    OUT_RESET
  }

  // 4.5.3 Save & load from/to binary
  TEST(Doc, user_guide_4_5_3_ex1) {
    OUT_REDIRECT
    #include "user_guide_4_5_3_ex1.cpp"
    OUT_RESET
  }

  // 5. Scalar
  // 5.1 Define/Declare a Scalar
  TEST(Doc, user_guide_5_1_ex1) {
    OUT_REDIRECT
    #include "user_guide_5_1_ex1.cpp"
    OUT_RESET
  }

  TEST(Doc, user_guide_5_1_ex2) {
    OUT_REDIRECT
    #include "user_guide_5_1_ex2.cpp"
    OUT_RESET
  }

  TEST(Doc, user_guide_5_1_ex3) {
    OUT_REDIRECT
    #include "user_guide_5_1_ex3.cpp"
    OUT_RESET
  }

  // 5.2 Change date type
  TEST(Doc, user_guide_5_2_ex1) {
    OUT_REDIRECT
    #include "user_guide_5_2_ex1.cpp"
    OUT_RESET
  }

  // 5.3 Application scenarios
  TEST(Doc, user_guide_5_3_ex1) {
    OUT_REDIRECT
    #include "user_guide_5_3_ex1.cpp"
    OUT_RESET
  }

  // 7. UniTensor
  // 7.4 Bond
  // 7.4.1 Symmetry object
  TEST(Doc, user_guide_7_4_1_ex1) {
    OUT_REDIRECT
    #include "user_guide_7_4_1_ex1.cpp"
    OUT_RESET
  }

  // 7.4.2 Creating Bonds with quantum numbers
  TEST(Doc, user_guide_7_4_2_ex1) {
    OUT_REDIRECT
    #include "user_guide_7_4_2_ex1.cpp"
    OUT_RESET
  }

  // 7.8 Get/set UniTensor element
  // 7.8.1 UniTensor without symmetries
  TEST(Doc, user_guide_7_8_1_ex1) {
    OUT_REDIRECT
    #include "user_guide_7_8_1_ex1.cpp"
    OUT_RESET
  }

  TEST(Doc, user_guide_7_8_1_ex2) {
    OUT_REDIRECT
    #include "user_guide_7_8_1_ex2.cpp"
    OUT_RESET
  }

  // 7.8.1 UniTensor with symmetries
  TEST(Doc, user_guide_7_8_2_ex1) {
    // need to add
  }

  // 8. Contraction
  // 8.1 Network
  // 8.1.2 Put UniTensors and Launch
  TEST(Doc, user_guide_8_1_2_ex1_2) {
    OUT_REDIRECT_FILE("user_guide_8_1_2_ex1")
    #include "user_guide_8_1_2_ex1.cpp"
    OUT_RESET
    #include "user_guide_8_1_2_ex2.cpp"
  }

  // 10. Iterative solver
  // 10.1 LinOp class
  TEST(Doc, user_guide_10_1_ex1) {
    OUT_REDIRECT
    #include "user_guide_10_1_ex1.cpp"
    OUT_RESET
  }

  // 10.1.2 Inherit the LinOp class
  TEST(Doc, user_guide_10_1_2_ex1) {
    OUT_REDIRECT
    #include "user_guide_10_1_2_ex1.cpp"
    OUT_RESET
  }

  // 10.2 Lanczos solver
  TEST(Doc, user_guide_10_2_ex1) {
    OUT_REDIRECT
    #include "user_guide_10_2_ex1.cpp"
    OUT_RESET
  }

}  // namespace DocTest
