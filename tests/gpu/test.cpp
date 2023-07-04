#include "cytnx.hpp"
#include <complex>

using namespace std;
using namespace cytnx;

typedef cytnx::Accessor ac;

int main(int argc, char *argv[]) {
  /*
  vector<cytnx_int64> A(10);
  for(int i=0;i<10;i++){
      A[i] = rand()%500;
  }
  cout << vec_unique(A) << endl;
  */

  // Device.Print_Property();
  cytnx_complex128 testC(1, 1);
  cytnx_complex64 testCf(1, 1);
  cout << pow(testCf, 2) << endl;
  cout << testC << endl;
  cout << pow(testC, 2) << endl;

  Tensor Dtst = arange(60).reshape({4, 3, 5});
  Tensor Dtst_s = Dtst[{ac(1), ac::all(), ac::range(1, 3)}];
  cout << Dtst << Dtst_s;
  Dtst = arange(10);
  Dtst.append(44);
  cout << Dtst << endl;
  return 0;

  Storage s1;
  s1.Init(0, Type.Double, Device.cpu);
  s1.set_zeros();
  for (int i = 0; i < 100; i++) {
    s1.append(i);
  }
  cout << s1 << endl;
  cout << s1.capacity() << endl;

  Storage svt;
  vector<cytnx_double> tmpVVd(30);
  for (int i = 0; i < tmpVVd.size(); i++) {
    cout << tmpVVd[i] << endl;
  }
  svt.from_vector(tmpVVd);
  cout << svt << endl;

  Tensor DD1 = arange(1., 5., 1.);
  cout << DD1 << endl;

  DD1.Save("test");
  Tensor DD2;
  DD2.Load("test.cytn");
  cout << DD2 << endl;

  Tensor DD = arange(1., 5., 1.);
  Tensor sDD = arange(0.4, 0.7, 0.1);
  cout << DD << sDD << endl;

  cout << linalg::Tridiag(DD, sDD, true);

  Tensor Trr({1, 1, 1}, Type.Double);
  Trr.reshape_({1, 1, 1, 1, -1});
  cout << Trr << endl;

  Tensor Xt1 = arange(4, Type.Double).reshape({2, 2});
  Tensor Xt2 = arange(12, Type.Double).reshape({4, 3});
  cout << Xt1 << endl;
  cout << Xt2 << endl;
  cout << linalg::Kron(Xt1, Xt2);

  return 0;

  Tensor A = arange(10, Type.Double);
  Tensor B = arange(0, 1, 0.1, Type.Double);
  cout << A << B << endl;
  Tensor C = linalg::Vectordot(A, B);
  cout << C.item<double>();

  A = zeros(4);
  A.at<double>({1}) = 1;
  A.at<double>({2}) = 1;
  B = zeros(4);
  B.at<double>({0}) = 1;
  B.at<double>({3}) = 1;

  A.reshape_({2, 2});
  B.reshape_({2, 2});
  cout << linalg::Kron(B, B) << endl;

  return 0;
  // return 0;

  Storage s;
  s.Init(12, Type.Double, Device.cpu);
  s.set_zeros();
  s.at<double>(4) = 3;
  cout << s << endl;
  Storage s2 = s;
  Storage s3 = s.clone();
  cout << is(s, s2) << is(s, s3) << endl;
  cout << (s == s2) << (s == s3) << endl;

  Tensor x;
  x = zeros({3, 4, 5}, Type.Double, Device.cpu);
  // Tensor x = zeros({3,4,5});
  cout << x << endl;
  x.fill(5.);
  cout << x << endl;
  Tensor b = x.clone();
  Tensor c = linalg::Add(1, x);
  Tensor d = c + c;
  Tensor e = d + 3;
  Tensor f = 3 + d;
  f += 3;
  f += 3.;
  // f+=3;
  // f+=f;
  cout << f << endl;

  Tensor g = c - c;
  Tensor h = g - 3.;
  Tensor i = 3. - g;
  // i-=3.;
  // i-=i;
  cout << i << endl;

  Tensor y = i.get({ac::all(), ac(2), ac::all()});
  cout << y << endl;

  Tensor a = zeros({2, 3}, Type.Double, Device.cpu);
  // Tensor a = zeros({2,3});
  a.at<double>({0, 0}) = 3;
  a.at<double>({0, 1}) = 2;
  a.at<double>({0, 2}) = 2;
  a.at<double>({1, 0}) = 2;
  a.at<double>({1, 1}) = 3;
  a.at<double>({1, 2}) = -2;

  vector<Tensor> out = linalg::Svd(a, false, false);
  cout << out[0];

  Tensor Zo = zeros(10, Type.Double, Device.cpu);
  Tensor Zo2 = zeros({3, 4});
  Tensor Zp = arange(10);
  Tensor Zc = arange(0.1, 0, -0.2, Type.ComplexDouble);
  cout << Zc << endl;

  Zp.reshape_({2, 5});
  cout << Zp << endl;
  Tensor tmp = Zp.get({ac::all(), ac::range(0, 2)});
  cout << tmp;
  Zp.set({ac::all(), ac::range(1, 3)}, tmp);
  cout << Zp;
  Zp.set({ac::all(), ac::range(1, 3)}, 4);
  cout << Zp;
  /*
      Bond bd_in = Bond(3,BD_KET,{{0,1,-1, 4},
                                  {0,2,-1,-4},
                                  {1,0, 2, 2}}
                                ,{Symmetry::Zn(2),
                                  Symmetry::Zn(3),
                                  Symmetry::U1(),
                                  Symmetry::U1()});

      cout << bd_in << endl;

      Bond bd_r = Bond(10);
      cout << bd_r << endl;

      Bond bd_l = Bond(10,BD_KET);
      cout << bd_l << endl;

      Bond bd_dqu1 = Bond(3, BD_BRA,{{0,1},{2,2},{3,4}});
      Bond bd_dqu2 = Bond(5, BD_BRA,{{0,1},{2,2},{3,4},{-2,-4},{-1,-2}});
      Bond bd_dqu3 = bd_dqu1.combineBond(bd_dqu2);
      bd_dqu3.set_type(BD_KET);
      Bond bd_dqu4 = Bond(6, BD_KET,{{0,1},{2,2},{3,4},{-2,-4},{-1,-2},{3,4}});
      cout << bd_dqu1 << endl;
     cout << bd_dqu2 << endl;
      cout << bd_dqu3 << endl;
      cout << bd_dqu4.getDegeneracy({3,4}) << endl;
      vector<vector<cytnx_int64> > comm24 =
     vec2d_intersect(bd_dqu2.qnums(),bd_dqu4.qnums(),false,false); for(cytnx_uint64
     i=0;i<comm24.size();i++){ cout << comm24[i] << endl;
      }
      std::vector<Bond> dbds = {bd_dqu3,bd_dqu4,bd_dqu1,bd_dqu2};

      CyTensor dut1(dbds,{},2);
      dut1.print_diagram(true);
      dut1.permute_({2,3,0,1});
      dut1.print_diagram(true);
      auto bcbs = dut1.getTotalQnums();
      cout << bcbs[0] << endl;
      cout << bcbs[1] << endl;
      return 0;

      Bond bd_1 = Bond(3);
      Bond bd_2 = Bond(5);
      Bond bd_3 = Bond(4);
      Bond bd_4 = bd_1.combineBond(bd_2);
      //std::cout << bd_4 << std::endl;

      std::vector<Bond> bds = {bd_1,bd_2,bd_3};
      std::vector<cytnx_int64> labels = {};

      CyTensor ut1(bds,{},2);
      CyTensor ut2 = ut1.clone();
      ut1.print_diagram();
      cout << ut1 << endl;
      ut1.combineBonds({2,0},true,false);
      ut1.print_diagram();
      ut2.print_diagram();
      ut2.set_label(2,-4);
      ut2.print_diagram();

      CyTensor ut3 = Contract(ut1,ut2);
      ut3.print_diagram();



      Tensor X1 = arange(100);
      X1.reshape_({2,5,2,5});
      Tensor X2 = X1.clone();
      X1.permute_({2,0,1,3});
      X2.permute_({0,2,3,1});
      cout << X1 << X2 << endl;

      Tensor X1c = X1.clone();
      Tensor X2c = X2.clone();
      X1c.contiguous_();
      X2c.contiguous_();


      cout << X1 + X2 << endl;
      cout << X1c + X2c << endl;

      Tensor Bb = ones({3,4,5},Type.Bool);
      cout << Bb << endl;

      cout << X1c << endl;
      cout << X1 << endl;
      cout << (X1c == X1) << endl;

      string sx1 = " test this ";
      string sx2 = " sttr: str;   ";
      string sx3 = " accr: tt,0,t,;    ";
      string sx4 = ":";

      string sx5 = "(  ((A,B),C,(D, E,F)),G )";
      //str_strip(sx1);
      //str_strip(sx2);
      //str_strip(sx3);
      //cout << str_strip(sx1) << "|" << endl;
      //cout << str_strip(sx2) << "|" << endl;
      //cout << str_strip(sx3) << "|" << endl;

      cout << str_split(sx1,false) << endl;
      cout << str_split(sx4,false,":") << endl;
      cout << str_findall(sx5,"(),") << endl;
      //cout << str_findall(sx5,"\w") << endl;
      //cout << ut1 << endl;

      //cout << ut2 << endl;


      CyTensor T1 = CyTensor(Tensor({2,3,4,5}),2);
      CyTensor T2 = CyTensor(Tensor({4,6,7,8}),3);
      CyTensor T3 = CyTensor(Tensor({5,6,7,2}),4);



      Network net;
      net.Fromfile("test.net");

      net.PutCyTensor("A",T1,false);
      net.PutCyTensor("B",T2,false);
      net.PutCyTensor("C",T3,false);


      net.Launch();

      vector<float> vd;
      vector<vector<double> > vvd;
      vector<vector<vector<double> > > vvvd;
      vector<vector<vector<vector<double> > > > vvvvd;

      cout << typeid(vd).name() << endl;
      cout << typeid(vvd).name() << endl;
      cout << typeid(vvvd).name() << endl;
      cout << typeid(vvvvd).name() << endl;
      Tensor x1 = arange(2*3*4);
      x1.reshape_({2,3,4});
      cout << x1 << endl;
      cout << x1.is_contiguous() << endl;

      x1.permute_({2,0,1});
      cout << x1 << endl;
      cout << x1.is_contiguous() << endl;
      x1.contiguous_();
      cout << x1 << endl;
      */
  // x1.reshape_({2,2,3,2});
  // cout << x1 << endl;

  // Tensor tmp = zeros({3,2,4});
  /*
  ut1.print_diagram(true);
  cout << ut1 << endl;
  Tensor sss = ut1.get_block(); // this will return a copied block

  cout << sss << endl;
  sss.at<double>({0,0,0}) = 3;
  cout << ut1 << endl;

  CyTensor re(sss,2); // construct by block will not copy, and share same memory.
  cout << re << endl;
  */

  return 0;
}
