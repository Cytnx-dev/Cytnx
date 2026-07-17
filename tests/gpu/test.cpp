#include <complex>

#include "cytnx.hpp"

namespace cytnx {
  namespace {

    typedef Accessor ac;

    int main(int argc, char *argv[]) {
      /*
      std::vector<cytnx_int64> A(10);
      for(int i=0;i<10;i++){
          A[i] = rand()%500;
      }
      std::cout << vec_unique(A) << std::endl;
      */

      // Device.Print_Property();
      cytnx_complex128 testC(1, 1);
      cytnx_complex64 testCf(1, 1);
      std::cout << std::pow(testCf, 2) << std::endl;
      std::cout << testC << std::endl;
      std::cout << std::pow(testC, 2) << std::endl;

      Tensor Dtst = arange(60).reshape({4, 3, 5});
      Tensor Dtst_s = Dtst[{ac(1), ac::all(), ac::range(1, 3)}];
      std::cout << Dtst << Dtst_s;
      Dtst = arange(10);
      Dtst.append(44);
      std::cout << Dtst << std::endl;
      return 0;

      Storage s1;
      s1.Init(0, Type.Double, Device.cpu);
      s1.set_zeros();
      for (int i = 0; i < 100; i++) {
        s1.append(i);
      }
      std::cout << s1 << std::endl;
      std::cout << s1.capacity() << std::endl;

      Storage svt;
      std::vector<cytnx_double> tmpVVd(30);
      for (int i = 0; i < tmpVVd.size(); i++) {
        std::cout << tmpVVd[i] << std::endl;
      }
      svt.from_vector(tmpVVd);
      std::cout << svt << std::endl;

      Tensor DD1 = arange(1., 5., 1.);
      std::cout << DD1 << std::endl;

      DD1.Save("test.cytn");
      Tensor DD2;
      DD2.Load("test.cytn");
      std::cout << DD2 << std::endl;

      Tensor DD = arange(1., 5., 1.);
      Tensor sDD = arange(0.4, 0.7, 0.1);
      std::cout << DD << sDD << std::endl;

      std::cout << linalg::Tridiag(DD, sDD, true);

      Tensor Trr({1, 1, 1}, Type.Double);
      Trr.reshape_({1, 1, 1, 1, -1});
      std::cout << Trr << std::endl;

      Tensor Xt1 = arange(4, Type.Double).reshape({2, 2});
      Tensor Xt2 = arange(12, Type.Double).reshape({4, 3});
      std::cout << Xt1 << std::endl;
      std::cout << Xt2 << std::endl;
      std::cout << linalg::Kron(Xt1, Xt2);

      return 0;

      Tensor A = arange(10, Type.Double);
      Tensor B = arange(0, 1, 0.1, Type.Double);
      std::cout << A << B << std::endl;
      Tensor C = linalg::Vectordot(A, B);
      std::cout << C.item<double>();

      A = zeros({4});
      A.at<double>({1}) = 1;
      A.at<double>({2}) = 1;
      B = zeros({4});
      B.at<double>({0}) = 1;
      B.at<double>({3}) = 1;

      A.reshape_({2, 2});
      B.reshape_({2, 2});
      std::cout << linalg::Kron(B, B) << std::endl;

      return 0;
      // return 0;

      Storage s;
      s.Init(12, Type.Double, Device.cpu);
      s.set_zeros();
      s.at<double>(4) = 3;
      std::cout << s << std::endl;
      Storage s2 = s;
      Storage s3 = s.clone();
      std::cout << is(s, s2) << is(s, s3) << std::endl;
      std::cout << (s == s2) << (s == s3) << std::endl;

      Tensor x;
      x = zeros({3, 4, 5}, Type.Double, Device.cpu);
      // Tensor x = zeros({3,4,5});
      std::cout << x << std::endl;
      x.fill(5.);
      std::cout << x << std::endl;
      Tensor b = x.clone();
      Tensor c = linalg::Add(1, x);
      Tensor d = c + c;
      Tensor e = d + 3;
      Tensor f = 3 + d;
      f += 3;
      f += 3.;
      // f+=3;
      // f+=f;
      std::cout << f << std::endl;

      Tensor g = c - c;
      Tensor h = g - 3.;
      Tensor i = 3. - g;
      // i-=3.;
      // i-=i;
      std::cout << i << std::endl;

      Tensor y = i.get({ac::all(), ac(2), ac::all()});
      std::cout << y << std::endl;

      Tensor a = zeros({2, 3}, Type.Double, Device.cpu);
      // Tensor a = zeros({2,3});
      a.at<double>({0, 0}) = 3;
      a.at<double>({0, 1}) = 2;
      a.at<double>({0, 2}) = 2;
      a.at<double>({1, 0}) = 2;
      a.at<double>({1, 1}) = 3;
      a.at<double>({1, 2}) = -2;

      std::vector<Tensor> out = linalg::Svd(a, false, false);
      std::cout << out[0];

      Tensor Zo = zeros({10}, Type.Double, Device.cpu);
      Tensor Zo2 = zeros({3, 4});
      Tensor Zp = arange(10);
      Tensor Zc = arange(0.1, 0, -0.2, Type.ComplexDouble);
      std::cout << Zc << std::endl;

      Zp.reshape_({2, 5});
      std::cout << Zp << std::endl;
      Tensor tmp = Zp.get({ac::all(), ac::range(0, 2)});
      std::cout << tmp;
      Zp.set({ac::all(), ac::range(1, 3)}, tmp);
      std::cout << Zp;
      Zp.set({ac::all(), ac::range(1, 3)}, 4);
      std::cout << Zp;
      /*
          Bond bd_in = Bond(3,BD_KET,{{0,1,-1, 4},
                                      {0,2,-1,-4},
                                      {1,0, 2, 2}}
                                    ,{Symmetry::Zn(2),
                                      Symmetry::Zn(3),
                                      Symmetry::U1(),
                                      Symmetry::U1()});

          std::cout << bd_in << std::endl;

          Bond bd_r = Bond(10);
          std::cout << bd_r << std::endl;

          Bond bd_l = Bond(10,BD_KET);
          std::cout << bd_l << std::endl;

          Bond bd_dqu1 = Bond(3, BD_BRA,{{0,1},{2,2},{3,4}});
          Bond bd_dqu2 = Bond(5, BD_BRA,{{0,1},{2,2},{3,4},{-2,-4},{-1,-2}});
          Bond bd_dqu3 = bd_dqu1.combineBond(bd_dqu2);
          bd_dqu3.set_type(BD_KET);
          Bond bd_dqu4 = Bond(6, BD_KET,{{0,1},{2,2},{3,4},{-2,-4},{-1,-2},{3,4}});
          std::cout << bd_dqu1 << std::endl;
         std::cout << bd_dqu2 << std::endl;
          std::cout << bd_dqu3 << std::endl;
          std::cout << bd_dqu4.getDegeneracy({3,4}) << std::endl;
          std::vector<std::vector<cytnx_int64> > comm24 =
         vec2d_intersect(bd_dqu2.qnums(),bd_dqu4.qnums(),false,false); for(cytnx_uint64
         i=0;i<comm24.size();i++){ std::cout << comm24[i] << std::endl;
          }
          std::vector<Bond> dbds = {bd_dqu3,bd_dqu4,bd_dqu1,bd_dqu2};

          CyTensor dut1(dbds,{},2);
          dut1.print_diagram(true);
          dut1.permute_({2,3,0,1});
          dut1.print_diagram(true);
          auto bcbs = dut1.getTotalQnums();
          std::cout << bcbs[0] << std::endl;
          std::cout << bcbs[1] << std::endl;
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
          std::cout << ut1 << std::endl;
          ut1.combineBonds({2,0},true,false);
          ut1.print_diagram();
          ut2.print_diagram();
          ut2.relabel_(2,-4);
          ut2.print_diagram();

          CyTensor ut3 = Contract(ut1,ut2);
          ut3.print_diagram();



          Tensor X1 = arange(100);
          X1.reshape_({2,5,2,5});
          Tensor X2 = X1.clone();
          X1.permute_({2,0,1,3});
          X2.permute_({0,2,3,1});
          std::cout << X1 << X2 << std::endl;

          Tensor X1c = X1.clone();
          Tensor X2c = X2.clone();
          X1c.contiguous_();
          X2c.contiguous_();


          std::cout << X1 + X2 << std::endl;
          std::cout << X1c + X2c << std::endl;

          Tensor Bb = ones({3,4,5},Type.Bool);
          std::cout << Bb << std::endl;

          std::cout << X1c << std::endl;
          std::cout << X1 << std::endl;
          std::cout << (X1c == X1) << std::endl;

          string sx1 = " test this ";
          string sx2 = " sttr: str;   ";
          string sx3 = " accr: tt,0,t,;    ";
          string sx4 = ":";

          string sx5 = "(  ((A,B),C,(D, E,F)),G )";
          //str_strip(sx1);
          //str_strip(sx2);
          //str_strip(sx3);
          //std::cout << str_strip(sx1) << "|" << std::endl;
          //std::cout << str_strip(sx2) << "|" << std::endl;
          //std::cout << str_strip(sx3) << "|" << std::endl;

          std::cout << str_split(sx1,false) << std::endl;
          std::cout << str_split(sx4,false,":") << std::endl;
          std::cout << str_findall(sx5,"(),") << std::endl;
          //std::cout << str_findall(sx5,"\w") << std::endl;
          //std::cout << ut1 << std::endl;

          //std::cout << ut2 << std::endl;


          CyTensor T1 = CyTensor(Tensor({2,3,4,5}),2);
          CyTensor T2 = CyTensor(Tensor({4,6,7,8}),3);
          CyTensor T3 = CyTensor(Tensor({5,6,7,2}),4);



          Network net;
          net.Fromfile("test.net");

          net.PutCyTensor("A",T1,false);
          net.PutCyTensor("B",T2,false);
          net.PutCyTensor("C",T3,false);


          net.Launch();

          std::vector<float> vd;
          std::vector<std::vector<double> > vvd;
          std::vector<std::vector<std::vector<double> > > vvvd;
          std::vector<std::vector<std::vector<std::vector<double> > > > vvvvd;

          std::cout << typeid(vd).name() << std::endl;
          std::cout << typeid(vvd).name() << std::endl;
          std::cout << typeid(vvvd).name() << std::endl;
          std::cout << typeid(vvvvd).name() << std::endl;
          Tensor x1 = arange(2*3*4);
          x1.reshape_({2,3,4});
          std::cout << x1 << std::endl;
          std::cout << x1.is_contiguous() << std::endl;

          x1.permute_({2,0,1});
          std::cout << x1 << std::endl;
          std::cout << x1.is_contiguous() << std::endl;
          x1.contiguous_();
          std::cout << x1 << std::endl;
          */
      // x1.reshape_({2,2,3,2});
      // std::cout << x1 << std::endl;

      // Tensor tmp = zeros({3,2,4});
      /*
      ut1.print_diagram(true);
      std::cout << ut1 << std::endl;
      Tensor sss = ut1.get_block(); // this will return a copied block

      std::cout << sss << std::endl;
      sss.at<double>({0,0,0}) = 3;
      std::cout << ut1 << std::endl;

      CyTensor re(sss,2); // construct by block will not copy, and share same memory.
      std::cout << re << std::endl;
      */

      return 0;
    }
  }  // namespace
}  // namespace cytnx
