#include "cytnx.hpp"
#include <gtest/gtest.h>

using namespace cytnx;
using namespace std;
using namespace testing;

namespace UserGuideTest {

// 1. Objects behavior
// 1.1. Everyting is reference
TEST(UserGuide, 1_1_ex1) {
  auto A = cytnx::Tensor({2,3});
  auto B = A;

  cout << cytnx::is(B,A) << endl;
}

TEST(UserGuide, 1_1_ex2) {
  auto A = cytnx::Tensor({2,3});
  auto B = A;
  auto C = A.clone();

  cout << cytnx::is(B,A) << endl;
  cout << cytnx::is(C,A) << endl;
}

// 1.2 clone
// 2. Device
// 2.1 Number of threads
TEST(UserGuide, 2_1_ex1) {
  cout << Device.Ncpus;
}

// 2.2 Number of GPUs
TEST(UserGuide, 2_2_ex1) {
  cout << Device.Ngpus;
}

// 3. Tensor
// 3.1 Creating a Tensor
// 3.1.1 Initialized Tensor
TEST(UserGuide, 3_1_1_ex1) {
  cytnx::Tensor A = cytnx::zeros({3,4,5});
}

TEST(UserGuide, 3_1_1_ex2) {
  auto A = cytnx::arange(10);     //rank-1 Tensor from [0,10) with step 1
  auto B = cytnx::arange(0,10,2); //rank-1 Tensor from [0,10) with step 2
  auto C = cytnx::ones({3,4,5});  //Tensor of shape (3,4,5) with all elements set to one.
  auto D = cytnx::eye(3);          //Tensor of shape (3,3) with diagonal elements set to one, all other entries are zero.
}

// 3.1.2 Random Tensor
TEST(UserGuide, 3_1_2_ex1) {
  auto A = cytnx::random::normal({3,4,5}, 0., 1.);    //Tensor of shape (3,4,5) with all elements distributed according
                                                      //to a normal distribution around 0 with standard deviation 1
  auto B = cytnx::random::uniform({3,4,5}, -1., 1.);  //Tensor of shape (3,4,5) with all elements distributed uniformly
                                                      //between -1 and 1
}

// 3.1.3 Tensor with different dtype and device
#ifdef UNI_GPU
TEST(UserGuide, 3_1_3_ex1) {
  auto A = cytnx::zeros({3,4,5},cytnx::Type.Int64,cytnx::Device.cuda);
}
#endif

// 3.1.4 Type conversion
TEST(UserGuide, 3_1_4_ex1) {
  auto A = cytnx::ones({3,4},cytnx::Type.Int64);
  auto B = A.astype(cytnx::Type.Double);
  cout << A.dtype_str() << endl;
  cout << B.dtype_str() << endl;
}

// 3.1.5 Transfer between devices
#ifdef UNI_GPU
TEST(UserGuide, 3_1_5_ex1) {
  auto A = cytnx::ones({2,2}); //on CPU
  auto B = A.to(cytnx::Device.cuda+0);
  cout << A << endl; // on CPU
  cout << B << endl; // on GPU

  A.to_(cytnx::Device.cuda);
  cout << A << endl; // on GPU
}
#endif

// 3.1.6 Tensor from Storage
TEST(UserGuide, 3_1_6_ex1) {
  // A & B share same memory
  auto A = cytnx::Storage(10);
  auto B = cytnx::Tensor::from_storage(A);

  // A & C have different memory
  auto C = cytnx::Tensor::from_storage(A.clone());
}

// 3.2 Manipulating Tensors
// 3.2.1 reshape
TEST(UserGuide, 3_2_1_ex1) {
  auto A = cytnx::arange(24);
  auto B = A.reshape(2,3,4);
  cout << A << endl;
  cout << B << endl;
}

TEST(UserGuide, 3_2_1_ex2) {
  auto A = cytnx::arange(24);
  cout << A << endl;
  A.reshape_(2,3,4);
  cout << A << endl;
}

// 3.2.2 permute
TEST(UserGuide, 3_2_2_ex1) {
  auto A = cytnx::arange(24).reshape(2,3,4);
  auto B = A.permute(1,2,0);
  cout << A << endl;
  cout << B << endl;
}

TEST(UserGuide, 3_2_2_ex2) {
  auto A = cytnx::arange(24).reshape(2,3,4);
  cout << A.is_contiguous() << endl;
  cout << A << endl;

  A.permute_(1,0,2);
  cout << A.is_contiguous() << endl;
  cout << A << endl;

  A.contiguous_();
  cout << A.is_contiguous() << endl;
}

// 3.3 Accessing elements
// 3.3.1 Get elements
TEST(UserGuide, 3_3_1_ex1) {
  auto A = cytnx::arange(24).reshape(2,3,4);
  cout << A << endl;

  auto B = A(0,":","1:4:2");
  cout << B << endl;

  auto C = A(":",1);
  cout << C << endl;
}

TEST(UserGuide, 3_3_1_ex2) {
  auto A = cytnx::arange(24).reshape(2,3,4);
  auto B = A(0,0,1);
  Scalar C = B.item();
  double Ct = B.item<double>();

  cout << B << endl;
  cout << C << endl;
  cout << Ct << endl;
}

// 3.3.2 Get elements
TEST(UserGuide, 3_3_2_ex1) {
  auto A = cytnx::arange(24).reshape(2,3,4);
  auto B = cytnx::zeros({3,2});
  cout << A << endl;
  cout << B << endl;

  A(1,":","::2") = B;
  cout << A << endl;

  A(0,"::2",2) = 4;
  cout << A << endl;
}

// 3.3.3 Low-level API (C++ only)
/*
TEST(UserGuide, 3_3_3_ex1) {
   typedef ac=cytnx::Accessor;

   ac(4);     // this is equal to index '4' in Python
   ac::all(); // this is equal to ':' in Python
   ac::range(0,4,2); // this is equal to '0:4:2' in Python
}

TEST(UserGuide, 3_3_3_ex2) {
  typedef ac=cytnx::Accessor;
  auto A = cytnx::arange(24).reshape(2,3,4);
  auto B = cytnx::zeros({3,2});

  // [get] this is equal to A[0,:,1:4:2] in Python:
  auto C = A[{ac(0},ac::all(),ac::range(1,4,2)}];

  // [set] this is equal to A[1,:,0:4:2] = B in Python:
  A[{ac(1),ac::all(),ac::range(0,4,2)}] = B;
}

TEST(UserGuide, 3_3_3_ex3) {
  typedef ac=cytnx::Accessor;
  auto A = cytnx::arange(24).reshape(2,3,4);
  auto B = cytnx::zeros({3,2});

  // [get] this is equal to A[0,:,1:4:2] in Python:
  auto C = A.get({ac(0},ac::all(),ac::range(1,4,2)});

  // [set] this is equal to A[1,:,0:4:2] = B in Python:
  A.set({ac(1),ac::all(),ac::range(0,4,2)}, B);
}
*/

// 3.4 Tensor arithmetic
// 3.4.2 Tensor-scalar arithmetic
TEST(UserGuide, 3_4_2_ex1) {
  auto A = cytnx::ones({3,4});
  cout << A << endl;

  auto B = A + 4;
  cout << B << endl;

  auto C = A - std::complex<double>(0,7); //type promotion
  cout << C << endl;
}

// 3.4.3 Tensor-Tensor arithmetic
TEST(UserGuide, 3_4_3_ex1) {
  auto A = cytnx::arange(12).reshape(3,4);
  cout << A << endl;

  auto B = cytnx::ones({3,4})*4;
  cout << B << endl;

  auto C = A * B;
  cout << C << endl;
}

// 3.4.4 Equivalent APIs (C++ only)
TEST(UserGuide, 3_4_4_ex1) {
  auto A = cytnx::ones({3,4});
  auto B = cytnx::arange(12).reshape(3,4);

  // these two are equivalent to C = A+B;
  auto C = A.Add(B);
  auto D = cytnx::linalg::Add(A,B);

  // this is equivalent to A+=B;
  A.Add_(B);
}

// 3.6 Appending elements
TEST(UserGuide, 3_6_ex1) {
  auto A = cytnx::ones(4);
  cout << A << endl;
  A.append(4);
  cout << A << endl;
}

TEST(UserGuide, 3_6_ex2) {
  auto A = cytnx::ones({3,4,5});
  auto B = cytnx::ones({4,5})*2;
  cout << A << endl;
  cout << B << endl;

  A.append(B);
  cout << A << endl;
}

// 3.7 Save/Load
// 3.7.1 Save a Tensor
TEST(UserGuide, 3_7_1_ex1) {
  auto A = cytnx::arange(12).reshape(3,4);
  A.Save("T1");
}

// 3.7.2 Load a Tensor
TEST(UserGuide, 3_7_2_ex1) {
  auto A = cytnx::Tensor::Load("T1.cytn");
  cout << A << endl;
}

// 3.8 When will data be copied?
// 3.8.1 Reference to & Copy of objects
TEST(UserGuide, 3_8_1_ex1) {
  auto A = cytnx::zeros({3,4,5});
  auto B = A;

  cout << is(B,A) << endl;
}

TEST(UserGuide, 3_8_1_ex2) {
  auto A = cytnx::zeros({3,4,5});
  auto B = A.clone();

  cout << is(B,A) << endl;
}

// 3.8.2 Permute
TEST(UserGuide, 3_8_2_ex1) {
//================================
  auto A = cytnx::zeros({2,3,4});
  auto B = A.permute(0,2,1);

  cout << A << endl;
  cout << B << endl;

  cout << is(B,A) << endl;
//================================
  A(0,0,0) = 300;

  cout << A << endl;
  cout << B << endl;
//================================
  cout << B.same_data(A) << endl;
//================================
}

// 3.8.2 Contiguous
TEST(UserGuide, 3_8_3_ex1) {
//================================
  auto A = cytnx::zeros({2,3,4});
  auto B = A.permute(0,2,1);

  cout << A.is_contiguous() << endl;
  cout << B.is_contiguous() << endl;
//================================
  auto C = B.contiguous();

  cout << C << endl;
  cout << C.is_contiguous() << endl;

  cout << C.same_data(B) << endl;
//================================
}

// 4. Storage
// 4.1 Creating a Storage
TEST(UserGuide, 4_1_ex1) {
  auto A = cytnx::Storage(10,cytnx::Type.Double,cytnx::Device.cpu);
  A.set_zeros();

  cout << A << endl;
}

// 4.1.1 Type conversion
TEST(UserGuide, 4_1_1_ex1) {
  auto A = cytnx::Storage(10);
  A.set_zeros();

  auto B = A.astype(cytnx::Type.ComplexDouble);

  cout << A << endl;
  cout << B << endl;
}

// 4.1.2 Transfer between devices
#ifdef UNI_GPU
TEST(UserGuide, 4_1_2_ex1) {
  auto A = cytnx::Storage(4);

  auto B = A.to(cytnx::Device.cuda);
  cout << A.device_str() << endl;
  cout << B.device_str() << endl;

  A.to_(cytnx::Device.cuda);
  cout << A.device_str() << endl;
}
#endif

// 4.1.3 Get Storage of Tensor
TEST(UserGuide, 4_1_3_ex1) {
  auto A = cytnx::arange(10).reshape(2,5);
  auto B = A.storage();

  cout << A << endl;
  cout << B << endl;
}

// 4.2 Accessing elements
TEST(UserGuide, 4_2_1_ex1) {
  auto A = cytnx::Storage(6);
  A.set_zeros();
  cout << A << endl;

  A.at<double>(4) = 4;
  cout << A << endl;
}

// 4.2.1 Get/Set elements
TEST(UserGuide, 4_2_1_ex2) {
  auto A = cytnx::Storage(6);
  cout << A << endl;

  Scalar elemt = A.at(4);
  cout << elemt << endl;

  A.at(4) = 4;
  cout << A << endl;
}

// 4.2.2 Get raw-pointer (C++ only)
TEST(UserGuide, 4_2_2_ex1) {
  auto A = cytnx::Storage(6);
  double *pA = A.data<double>();
}

TEST(UserGuide, 4_2_2_ex2) {
  auto A = cytnx::Storage(6);
  void *pA = A.data();
}

// 4.3 Increase size
// 4.3.1 append
TEST(UserGuide, 4_3_1_ex1) {
  auto A = cytnx::Storage(4);
  A.set_zeros();
  cout << A << endl;

  A.append(500);
  cout << A << endl;
}

// 4.3.1 resize
TEST(UserGuide, 4_3_2_ex1) {
  auto A = cytnx::Storage(4);
  cout << A.size() << endl;

  A.resize(5);
  cout << A.size() << endl;
}

// 4.4 From/To C++ .vector
#ifdef UNI_GPU
TEST(UserGuide, 4_4_ex1) {
  vector<double> vA(4,6);

  auto A = cytnx::Storage::from_vector(vA);
  auto B = cytnx::Storage::from_vector(vA,cytnx::Device.cuda);

  cout << A << endl;
  cout << B << endl;
}
#endif

TEST(UserGuide, 4_4_ex2) {
  Storage sA = {3.,4.,5.,6.};

  print(sA.dtype_str());

  auto vA = sA.vector<double>();

  print(vA);
}

// 4.5 Save/Load
// 4.5.1 Save a Storage
TEST(UserGuide, 4_5_1_ex1) {
  auto A = cytnx::Storage(4);
  A.fill(6);
  A.Save("S1");
}

// 4.5.2 Load a Storage
TEST(UserGuide, 4_5_2_ex1) {
  auto A = cytnx::Storage::Load("S1.cyst");
  cout << A << endl;
}

// 4.5.3 Save & load from/to binary
TEST(UserGuide, 4_5_3_ex1) {
  // read
  auto A = cytnx::Storage(10);
  A.fill(10);
  cout << A << endl;

  A.Tofile("S1");

  //load
  auto B = cytnx::Storage::Fromfile("S1",cytnx::Type.Double);

  cout << B << endl;
}

// 5. Scalar
// 5.1 Define/Declare a Scalar
TEST(UserGuide, 5_1_ex1) {
  double cA = 1.33;
  Scalar A(cA);
  cout << A << endl;
}

TEST(UserGuide, 5_1_ex2) {
  Scalar A(double(1.33));

  Scalar A2 = double(1.33);

  Scalar A3(10,Type.Double);

  cout << A << A2 << A3 << endl;
}

TEST(UserGuide, 5_1_ex3) {
  Scalar A = 10;
  cout << A << endl;

  auto fA = float(A); // convert to float
  cout << typeid(fA).name() << fA << endl;

  // convert to complex double
  auto cdA = complex128(A);
  cout << cdA << endl;

  // convert to complex float
  auto cfA = complex64(A);
  cout << cfA << endl;
}

// 5.2 Change date type 
TEST(UserGuide, 5_2_ex1) {
  Scalar A(1.33);
  cout << A << endl;

  A = A.astype(Type.Float);
  cout << A << endl;
}

// 5.3 Application scenarios
TEST(UserGuide, 5_3_ex1) {
  vector<Scalar> out;

  out.push_back(Scalar(1.33)); //double
  out.push_back(Scalar(10));   //int
  out.push_back(Scalar(cytnx_complex128(3,4))); //complex double

  cout << out[0] << out[1] << out[2] << endl;
}

// 7. UniTensor
// 7.4 Bond
// 7.4.1 Symmetry object
TEST(UserGuide, 7_4_1_ex1) {
  Symmetry sym_u1 = cytnx::Symmetry::U1();
  Symmetry sym_z2 = cytnx::Symmetry::Zn(2);
  
  cout << sym_u1 << endl;
  cout << sym_z2 << endl;
}

// 7.4.2 Creating Bonds with quantum numbers
TEST(UserGuide, 7_4_2_ex1) {
  // This creates an KET (IN) Bond with quantum number 0,-4,-2,3 with degs 3,4,3,2 respectively.
  Bond bd_sym_u1_a = cytnx::Bond(cytnx::BD_KET,
                                 {cytnx::Qs(0)>>3,cytnx::Qs(-4)>>4,cytnx::Qs(-2)>>3,cytnx::Qs(3)>>2},
                                 {cytnx::Symmetry::U1()});
  
  // equivalent:
  bd_sym_u1_a = cytnx::Bond(cytnx::BD_IN,
                            {cytnx::Qs(0),cytnx::Qs(-4),cytnx::Qs(-2),cytnx::Qs(3)},
                            {3,4,3,2},{cytnx::Symmetry::U1()});
  
  print(bd_sym_u1_a);
}

TEST(UserGuide, 7_4_2_ex2) {
  auto bd_sym_u1z2_a = cytnx::Bond(cytnx::BD_KET,
                                   {cytnx::Qs(0 ,0)>>3,
                                    cytnx::Qs(-4,1)>>4,
                                    cytnx::Qs(-2,0)>>3,
                                    cytnx::Qs(3 ,1)>>2},
                                   {cytnx::Symmetry::U1(),cytnx::Symmetry::Zn(2)});
  
  print(bd_sym_u1z2_a);
}

// 7.8 Get/set UniTensor element
// 7.8.1 UniTensor without symmetries
TEST(UserGuide, 7_8_1_ex1) {
  auto T = cytnx::UniTensor(cytnx::arange(9).reshape(3,3));
  print(T.at({0,2}));
}

TEST(UserGuide, 7_8_1_ex2) {
  auto T = cytnx::UniTensor(cytnx::arange(9).reshape(3,3));
  print(T.at({0,2}));
  T.at({0,2}) = 7;
  print(T.at({0,2}));
}

// 7.8.1 UniTensor with symmetries
TEST(UserGuide, 7_8_2_ex1) {
  //need to add
}

// 8. Contraction
// 8.1 Network
// 8.1.2 Put UniTensors and Launch
TEST(UserGuide, 8_1_2_ex1) {
  // initialize tensors
  auto w = cytnx::UniTensor(cytnx::random::normal({2,2,2,2}, 0., 1.));
  auto c0 = cytnx::UniTensor(cytnx::random::normal({8,8}, 0., 1.));
  auto c1 = cytnx::UniTensor(cytnx::random::normal({8,8}, 0., 1.));
  auto c2 = cytnx::UniTensor(cytnx::random::normal({8,8}, 0., 1.));
  auto c3 = cytnx::UniTensor(cytnx::random::normal({8,8}, 0., 1.));
  auto t0 = cytnx::UniTensor(cytnx::random::normal({8,2,8}, 0., 1.));
  auto t1 = cytnx::UniTensor(cytnx::random::normal({8,2,8}, 0., 1.));
  auto t2 = cytnx::UniTensor(cytnx::random::normal({8,2,8}, 0., 1.));
  auto t3 = cytnx::UniTensor(cytnx::random::normal({8,2,8}, 0., 1.));
  
  // initialize network object from ctm.net file
  Network net = cytnx::Network("ctm.net");
  
  // put tensors
  net.PutUniTensor("w",w);
  net.PutUniTensor("c0",c0);
  net.PutUniTensor("c1",c1);
  net.PutUniTensor("c2",c2);
  net.PutUniTensor("c3",c3);
  net.PutUniTensor("t0",t0);
  net.PutUniTensor("t1",t1);
  net.PutUniTensor("t2",t2);
  net.PutUniTensor("t3",t3);
  
  cout << net;
//================================
  UniTensor Res = net.Launch(true);
}

TEST(UserGuide, 10_1_ex1) {
  auto x = cytnx::ones(4);
  auto H = cytnx::arange(16).reshape(4,4);
  
  auto y = cytnx::linalg::Dot(H,x);
  
  cout << x << endl;
  cout << H << endl;
  cout << y << endl;
}

// 10. Iterative solver
// 10.1 LinOp class
// 10.1.1 Pass a function
//10_1_1_ex1
Tensor myfunc(const Tensor &v){
    Tensor out = v.clone();
    out(0) = v(3); //swap
    out(3) = v(0); //swap
    out(1)+=1; //add 1
    out(2)+=1; //add 1
    return out;
}
TEST(UserGuide, 10_1_1_ex1) {
//need add
}

TEST(UserGuide, 10_1_2_ex1) {
  using namespace cytnx;
  class MyOp: public LinOp{
      public:
          double AddConst;
  
          MyOp(double aconst):
              LinOp("mv",4,Type.Double,Device.cpu){ //invoke base class constructor!
  
              this->AddConst = aconst;
          }
  
          Tensor matvec(const Tensor& v) override{
              auto out = v.clone();
              out(0) = v(3); //swap
              out(3) = v(0); //swap
              out(1)+=this->AddConst; //add const
              out(2)+=this->AddConst; //add const
              return out;
          }
  
  };
  auto myop = MyOp(7);
  auto x = cytnx::arange(4);
  auto y = myop.matvec(x);
  
  cout << x << endl;
  cout << y << endl;
}

// 10.2 Lanczos solver
TEST(UserGuide, 10_2_ex1) {
  using namespace cytnx;
  class MyOp: public LinOp{
      public:
      MyOp(): LinOp("mv",4){}
  
      private:
      Tensor matvec(const Tensor &v) override{
          auto A = arange(16).reshape(4,4);
          A += A.permute(1,0);
          return linalg::Dot(A,v);
      }
  
  };
  
  auto op = MyOp();
  
  auto v0 = arange(4); // trial state
  auto ev = linalg::Lanczos_ER(&op, 1, true, 10000, 1.0e-14, false, v0, 3);
  
  cout << ev[0] << endl; //eigenval
  cout << ev[1] << endl; //eigenvec
}

} //namespace
