#include "cytnx.hpp"
#include <gtest/gtest.h>

//redirect standard output: if you need to output to another file, you can use these
//    macro to redirect the standart output
#define OUT_REDIRECT_BASE \
    coutbuf = cout.rdbuf(); \
    cout.rdbuf(output_file_o.rdbuf()); //redirect std::cout to file

#define OUT_REDIRECT \
	string suite_name = testing::UnitTest::GetInstance()->current_test_info()->name();\ 
	output_file_o.open(output_dir + suite_name + ".out");\
	OUT_REDIRECT_BASE

#define OUT_REDIRECT_FILE(OUT_FILE) \
	output_file_o.open(output_dir + OUT_FILE + ".out");\
	OUT_REDIRECT_BASE

#define OUT_RESET cout.rdbuf(coutbuf); \
	output_file_o.close();

using namespace cytnx;
using namespace std;
using namespace testing;

namespace UserGuideTest {
const string output_dir = "../../tests/user_guide/outputs/";

//output redirect
streambuf *coutbuf = nullptr;
ofstream output_file_o;

// 1. Objects behavior
// 1.1. Everyting is reference
TEST(UserGuide, 1_1_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/1_1_ex1.cpp"
  OUT_RESET
}

TEST(UserGuide, 1_2_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/1_2_ex1.cpp"
  OUT_RESET
}

// 1.2 clone
// 2. Device
// 2.1 Number of threads
TEST(UserGuide, 2_1_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/2_1_ex1.cpp"
  OUT_RESET
}

// 2.2 Number of GPUs
TEST(UserGuide, 2_2_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/2_2_ex1.cpp"
  OUT_RESET
}

// 3. Tensor
// 3.1 Creating a Tensor
// 3.1.1 Initialized Tensor
TEST(UserGuide, 3_1_1_ex1) {
  #include "guide_codes/3_1_1_ex1.cpp"
}

TEST(UserGuide, 3_1_1_ex2) {
  #include "guide_codes/3_1_1_ex2.cpp"
}

// 3.1.2 Random Tensor
TEST(UserGuide, 3_1_2_ex1) {
  #include "guide_codes/3_1_2_ex1.cpp"
}

// 3.1.3 Tensor with different dtype and device
#ifdef UNI_GPU
TEST(UserGuide, 3_1_3_ex1) {
  #include "guide_codes/3_1_3_ex1.cpp"
}
#endif

// 3.1.4 Type conversion
TEST(UserGuide, 3_1_4_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/3_1_4_ex1.cpp"
  OUT_RESET
}

// 3.1.5 Transfer between devices
#ifdef UNI_GPU
TEST(UserGuide, 3_1_5_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/3_1_5_ex1.cpp"
  OUT_RESET
}
#endif

// 3.1.6 Tensor from Storage
TEST(UserGuide, 3_1_6_ex1) {
  #include "guide_codes/3_1_6_ex1.cpp"
}

// 3.2 Manipulating Tensors
// 3.2.1 reshape
TEST(UserGuide, 3_2_1_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/3_2_1_ex1.cpp"
  OUT_RESET
}

TEST(UserGuide, 3_2_1_ex2) {
  OUT_REDIRECT 
  #include "guide_codes/3_2_1_ex2.cpp"
  OUT_RESET
}

// 3.2.2 permute
TEST(UserGuide, 3_2_2_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/3_2_2_ex1.cpp"
  OUT_RESET
}

TEST(UserGuide, 3_2_2_ex2) {
  OUT_REDIRECT 
  #include "guide_codes/3_2_1_ex2.cpp"
  OUT_RESET
}

// 3.3 Accessing elements
// 3.3.1 Get elements
TEST(UserGuide, 3_3_1_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/3_3_1_ex1.cpp"
  OUT_RESET
}

TEST(UserGuide, 3_3_1_ex2) {
  OUT_REDIRECT 
  #include "guide_codes/3_3_1_ex2.cpp"
  OUT_RESET
}

// 3.3.2 Get elements
TEST(UserGuide, 3_3_2_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/3_3_2_ex1.cpp"
  OUT_RESET
}

// 3.3.3 Low-level API (C++ only)
/*
TEST(UserGuide, 3_3_3_ex1) {
  #include "guide_codes/3_3_3_ex1.cpp"
}

TEST(UserGuide, 3_3_3_ex2) {
  #include "guide_codes/3_3_3_ex2.cpp"
}

TEST(UserGuide, 3_3_3_ex3) {
}
  #include "guide_codes/3_3_3_ex3.cpp"
*/

// 3.4 Tensor arithmetic
// 3.4.2 Tensor-scalar arithmetic
TEST(UserGuide, 3_4_2_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/3_4_2_ex1.cpp"
  OUT_RESET
}

// 3.4.3 Tensor-Tensor arithmetic
TEST(UserGuide, 3_4_3_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/3_4_3_ex1.cpp"
  OUT_RESET
}

// 3.4.4 Equivalent APIs (C++ only)
TEST(UserGuide, 3_4_4_ex1) {
  #include "guide_codes/3_4_4_ex1.cpp"
}

// 3.6 Appending elements
TEST(UserGuide, 3_6_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/3_6_ex1.cpp"
  OUT_RESET
}

TEST(UserGuide, 3_6_ex2) {
  OUT_REDIRECT 
  #include "guide_codes/3_6_ex2.cpp"
  OUT_RESET
}

// 3.7 Save/Load
// 3.7.1 Save a Tensor
TEST(UserGuide, 3_7_1_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/3_7_1_ex1.cpp"
  OUT_RESET
}

// 3.7.2 Load a Tensor
TEST(UserGuide, 3_7_2_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/3_7_2_ex1.cpp"
  OUT_RESET
}

// 3.8 When will data be copied?
// 3.8.1 Reference to & Copy of objects
TEST(UserGuide, 3_8_1_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/3_8_1_ex1.cpp"
  OUT_RESET
}

TEST(UserGuide, 3_8_1_ex2) {
  OUT_REDIRECT 
  #include "guide_codes/3_8_1_ex2.cpp"
  OUT_RESET
}

// 3.8.2 Permute
TEST(UserGuide, 3_8_2_ex1_2_3) {
  OUT_REDIRECT_FILE("3_8_2_ex1") 
  #include "guide_codes/3_8_2_ex1.cpp"
  OUT_RESET
  OUT_REDIRECT_FILE("3_8_2_ex2") 
  #include "guide_codes/3_8_2_ex2.cpp"
  OUT_RESET
  OUT_REDIRECT_FILE("3_8_2_ex3") 
  #include "guide_codes/3_8_2_ex3.cpp"
  OUT_RESET
}

// 3.8.2 Contiguous
TEST(UserGuide, 3_8_3_ex1_2) {
  OUT_REDIRECT_FILE("3_8_3_ex1") 
  #include "guide_codes/3_8_3_ex1.cpp"
  OUT_RESET
  OUT_REDIRECT_FILE("3_8_3_ex2") 
  #include "guide_codes/3_8_3_ex2.cpp"
  OUT_RESET
}

// 4. Storage
// 4.1 Creating a Storage
TEST(UserGuide, 4_1_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/4_1_ex1.cpp"
  OUT_RESET
}

// 4.1.1 Type conversion
TEST(UserGuide, 4_1_1_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/4_1_1_ex1.cpp"
  OUT_RESET
}

// 4.1.2 Transfer between devices
#ifdef UNI_GPU
TEST(UserGuide, 4_1_2_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/4_1_2_ex1.cpp"
  OUT_RESET
}
#endif

// 4.1.3 Get Storage of Tensor
TEST(UserGuide, 4_1_3_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/4_1_3_ex1.cpp"
  OUT_RESET
}

// 4.2 Accessing elements
// 4.2.1 Get/Set elements
TEST(UserGuide, 4_2_1_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/4_2_1_ex1.cpp"
  OUT_RESET
}

TEST(UserGuide, 4_2_1_ex2) {
  OUT_REDIRECT 
  #include "guide_codes/4_2_1_ex2.cpp"
  OUT_RESET
}

// 4.2.2 Get raw-pointer (C++ only)
TEST(UserGuide, 4_2_2_ex1) {
  #include "guide_codes/4_2_2_ex1.cpp"
}

TEST(UserGuide, 4_2_2_ex2) {
  #include "guide_codes/4_2_2_ex2.cpp"
}

// 4.3 Increase size
// 4.3.1 append
TEST(UserGuide, 4_3_1_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/4_3_1_ex1.cpp"
  OUT_RESET
}

// 4.3.2 resize
TEST(UserGuide, 4_3_2_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/4_3_2_ex1.cpp"
  OUT_RESET
}

// 4.4 From/To C++ .vector
#ifdef UNI_GPU
TEST(UserGuide, 4_4_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/4_4_ex1.cpp"
  OUT_RESET
}
#endif

TEST(UserGuide, 4_4_ex2) {
  OUT_REDIRECT 
  #include "guide_codes/4_4_ex2.cpp"
  OUT_RESET
}

// 4.5 Save/Load
// 4.5.1 Save a Storage
TEST(UserGuide, 4_5_1_ex1) {
  #include "guide_codes/4_5_1_ex1.cpp"
}

// 4.5.2 Load a Storage
TEST(UserGuide, 4_5_2_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/4_5_2_ex1.cpp"
  OUT_RESET
}

// 4.5.3 Save & load from/to binary
TEST(UserGuide, 4_5_3_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/4_5_3_ex1.cpp"
  OUT_RESET
}

// 5. Scalar
// 5.1 Define/Declare a Scalar
TEST(UserGuide, 5_1_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/5_1_ex1.cpp"
  OUT_RESET
}

TEST(UserGuide, 5_1_ex2) {
  OUT_REDIRECT 
  #include "guide_codes/5_1_ex2.cpp"
  OUT_RESET
}

TEST(UserGuide, 5_1_ex3) {
  OUT_REDIRECT 
  #include "guide_codes/5_1_ex3.cpp"
  OUT_RESET
}

// 5.2 Change date type 
TEST(UserGuide, 5_2_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/5_2_ex1.cpp"
  OUT_RESET
}

// 5.3 Application scenarios
TEST(UserGuide, 5_3_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/5_3_ex1.cpp"
  OUT_RESET
}

// 7. UniTensor
// 7.4 Bond
// 7.4.1 Symmetry object
TEST(UserGuide, 7_4_1_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/7_4_1_ex1.cpp"
  OUT_RESET
}

// 7.4.2 Creating Bonds with quantum numbers
TEST(UserGuide, 7_4_2_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/7_4_2_ex1.cpp"
  OUT_RESET
}

// 7.8 Get/set UniTensor element
// 7.8.1 UniTensor without symmetries
TEST(UserGuide, 7_8_1_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/7_8_1_ex1.cpp"
  OUT_RESET
}

TEST(UserGuide, 7_8_1_ex2) {
  OUT_REDIRECT 
  #include "guide_codes/7_8_1_ex2.cpp"
  OUT_RESET
}

// 7.8.1 UniTensor with symmetries
TEST(UserGuide, 7_8_2_ex1) {
  //need to add
}

// 8. Contraction
// 8.1 Network
// 8.1.2 Put UniTensors and Launch
TEST(UserGuide, 8_1_2_ex1_2) {
  OUT_REDIRECT_FILE("8_1_2_ex1")
  #include "guide_codes/8_1_2_ex1.cpp"
  OUT_RESET
  #include "guide_codes/8_1_2_ex2.cpp"
}

// 10. Iterative solver
// 10.1 LinOp class
TEST(UserGuide, 10_1_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/10_1_ex1.cpp"
  OUT_RESET
}

// 10.1.2 Inherit the LinOp class
TEST(UserGuide, 10_1_2_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/10_1_2_ex1.cpp"
  OUT_RESET
}

// 10.2 Lanczos solver
TEST(UserGuide, 10_2_ex1) {
  OUT_REDIRECT 
  #include "guide_codes/10_2_ex1.cpp"
  OUT_RESET
}

} //namespace
