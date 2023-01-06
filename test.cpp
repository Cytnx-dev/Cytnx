#include "cytnx.hpp"
#include <complex>

using namespace std;
using namespace cytnx;

typedef cytnx::Accessor ac;

int main(int argc, char *argv[]) {

  auto ottt = linalg::Svd(arange(200).reshape(10,20));

  cout << ottt[0] << endl;

  Bond phy = Bond(BD_IN, {Qs(0), Qs(1)}, {1, 1});
  Bond aux = Bond(BD_IN, {Qs(1)}, {1});

  auto Sp = UniTensor({phy, phy.redirect(), aux},{0,2,-1});
  auto Sm = UniTensor({phy, phy.redirect(), aux.redirect()}, {1, 3, -1});
  auto Sz = UniTensor({phy, phy.redirect()});

  // Sp.get_block_({0,1,0}).item() = 1;

  Sp.at({0, 1, 0}) = 1;
  Sm.at({1, 0, 0}) = 1;
  Sz.at({0, 0}) = 1;
  Sz.at({1, 1}) = -1;

  auto PM = Sp.contract(Sm);
  auto ZZ = Sz.contract(Sz.relabels({"a", "b"}));

  PM.print_diagram();
  PM.print_blocks(true);

  auto MP = Sm.contract(Sp);

  MP.print_diagram();
  MP.print_blocks(true);

  auto Hpmmp = 0.5 * (PM + MP) + ZZ;
  Hpmmp.permute_({0,2,1,3});

  Hpmmp.print_blocks(true);

  Hpmmp.set_rowrank(2);
    
  Hpmmp.print_diagram();

  auto Exp_Hpmmp = linalg::ExpH(Hpmmp,1);

  Exp_Hpmmp.print_diagram();
  Exp_Hpmmp.print_blocks(true);

  auto Outsvd = linalg::Svd_truncate(Hpmmp,100, 0);
  /*
  Outsvd[0].print_diagram();
  Outsvd[0].print_blocks();
  Outsvd[1].print_diagram();
  Outsvd[1].print_blocks();
  Outsvd[2].print_diagram();
  Outsvd[2].print_blocks();
  */

  return 0;

  /*
  auto TNtst = arange(180).reshape(15,12);
  cout << TNtst << endl;
  cout << algo::Vsplit(TNtst,{2,5,4,3,1}) << endl;


  cout << algo::Hsplit(TNtst,{2,3,1,2,4}) << endl;

  return 0;
  */


  auto S = UniTensor(arange(200).reshape(4,5,2,5));

  S.print_diagram();
  S.print_blocks();

  S.combineBonds({3,1});

  S.print_diagram();
  S.print_blocks();


  std::vector<int> tmptt = {0,1,2,3,4,5,6,7};
  memcpy(&tmptt[0],&tmptt[1],sizeof(int)*(tmptt.size()-1));
  print(tmptt); 

  auto BBA = Bond(BD_KET, {Qs(0), Qs(1), Qs(0), Qs(1), Qs(2)}, {1, 2, 3, 4, 5});
  auto BBB = Bond(BD_KET, {Qs(-1), Qs(-1), Qs(0), Qs(2), Qs(1)}, {1, 2, 3, 4, 5});

  //auto Ttrace = UniTensor({BBA,BBA.redirect()});

  //Ttrace.Trace_(0,1);

  //Ttrace.print_diagram();
  //return 0;

 
  auto bba = BBA.clone();
  bba.group_duplicates_();
  auto bbb = BBB.clone();
  bbb.group_duplicates_();

  /*
  auto TAT = UniTensor({BBA, BBA, BBB.redirect(), BBB.redirect()});
  print(BBA);
  print(BBA);
  print(BBB.redirect());
  print(BBB.redirect());
  TAT.print_diagram(true);


  auto bbbrbba = bbb.redirect().clone();
  bbbrbba._impl->force_combineBond_(bba._impl,false);
  
  auto bbabbbr = bba.clone();
  bbabbbr._impl->force_combineBond_(bbb.redirect()._impl,false);

  auto tst = UniTensor({bba, bbb.redirect(), bbbrbba});
  auto tst2 = UniTensor({bba, bbabbbr,bbb.redirect()});
  //print("-------");
  //print(((BlockUniTensor*)tst._impl.get())->_inner_to_outer_idx);
  //print("-------");
  //print("-------");
  //print(((BlockUniTensor*)tst2._impl.get())->_inner_to_outer_idx);
  //print("-------");
  */
  print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^");

  auto tat = UniTensor({bba, bba, bbb.redirect(), bbb.redirect()});

  tat.print_diagram(true);

  auto Osvd = linalg::Svd(tat,true,true);
  
  Osvd[0].print_diagram();
  Osvd[0].print_blocks(false);
  Osvd[1].print_diagram();
  Osvd[1].print_blocks(false);
  Osvd[2].print_diagram();
  Osvd[2].print_blocks(false);

  return 0;

  Tensor Taa = arange(20).reshape(4, 5);
  Tensor Tbb = arange(12).reshape(4, 3);

  auto Too = linalg::Directsum(Taa, Tbb, {0});

  print(Too);

  return 0;
  auto tbB = Bond(BD_BRA, {Qs(0), Qs(2), Qs(-3), Qs(0), Qs(1), Qs(2)}, {2, 3, 4, 5, 6, 7});

  cout << tbB << endl;

  print(tbB.group_duplicates_());

  cout << tbB << endl;

  // std::vector<cytnx_int64>va = {1,0,9,7,2};
  // auto mapPer = vec_sort(va,true);

  // print(va);

  return 0;
  std::vector<int> avv = {0, 2, 3, 4};
  std::vector<int> bvv = {0, 3, 1, 4};
  std::vector<int> cvv = {1, 1, 1, 4};

  cout << (avv < bvv) << endl;
  cout << (avv > bvv) << endl;
  cout << (avv == bvv) << endl;

  cout << (bvv < cvv) << endl;
  cout << (bvv > cvv) << endl;
  cout << (bvv == cvv) << endl;

  cout << (cvv < avv) << endl;
  cout << (cvv > avv) << endl;
  cout << (cvv == avv) << endl;

  return 0;
  /*
  vector<cytnx_int64> A(10);
  for(int i=0;i<10;i++){
      A[i] = rand()%500;
  }
  cout << vec_unique(A) << endl;
  */

  /*
  auto TTNdir = UniTensor(arange(24).reshape(2,3,4)).relabels({"good","evil","badass"});

  TTNdir.print_diagram();
  TTNdir.print_blocks(false);

  TTNdir.tag();

  TTNdir.print_diagram();
  TTNdir.print_blocks(false);
  Tensor A1 = arange(12).reshape(4,3).permute(1,0);
  Tensor A2 = arange(20).reshape(5,4);
  Tensor A3 = arange(32).reshape(8,4);
  print(A1);
  print(A2);
  print(A3);
  print(algo::Vstack({A1,A2,A3}));


  A1.permute_(1,0);
  A2.permute_(1,0);
  A3.permute_(1,0);

  print(algo::Hstack({A1,A2,A3}));

  */

  return 0;
  Bond B1 = Bond(BD_IN, {Qs(0), Qs(1)}, {3, 4});
  Bond B2 = Bond(BD_IN, {Qs(0), Qs(1)}, {5, 6});
  Bond B3 = Bond(BD_OUT, {Qs(0), Qs(1)}, {2, 3});
  Bond B4 = Bond(BD_OUT, {Qs(0), Qs(1)}, {7, 1});

  auto UTB = UniTensor({B1, B2, B3, B4});

  UTB.print_diagram();
  UTB.print_blocks(false);

  /*
  Sp.print_diagram(true);
  Sp.print_blocks(false);

  print(Sp.elem_exists({0,0,0}));
  print(Sp.elem_exists({0,1,0}));
  print(Sp.elem_exists({1,0,0}));
  print(Sp.elem_exists({1,1,0}));

  for(int i=0;i<2;i++)
    for(int j=0;j<2;j++){
        auto tmp = Sp.at({i,j,0});
        if(tmp.exists()) tmp = 1;
    }

  Sp.print_diagram(true);
  Sp.print_blocks(true);
  */

  return 0;

  // Bond bd_sym_s = Bond(BD_REG, {{0, 2}, {3, 5}, {1, 6}, {4, 1}}, {4, 7, 2, 3});
  Bond bd_sym_s = Bond(BD_KET, {{0}, {1}}, {4, 7});
  Bond bd_sym_s2 = Bond(BD_BRA, {{0}, {1}, {2}}, {8, 9, 3});
  Bond bd_sym_s3 = Bond(BD_BRA, {{-1}, {1}, {0}}, {2, 6, 5});

  Bond bd_sym_a =
    Bond(BD_KET, {{0, 0}, {1, 1}, {2, 1}, {3, 0}}, {4, 7, 2, 3}, {Symmetry::U1(), Symmetry::Zn(2)});
  /*
  UniTensor TTT({bd_sym_a,bd_sym_a,bd_sym_a.redirect(),bd_sym_a.redirect()},{1000,2000,3020,4024});
  TTT.print_diagram();
  TTT.print_blocks(false);


  auto TTTp = TTT.permute({1,3,0,2});
  TTTp.print_diagram();
  TTTp.print_blocks(false);
  TTTp.contiguous_();
  TTTp.print_blocks(false);
  */
  // UniTensor TTT({bd_sym_a,bd_sym_a.redirect(),bd_sym_a,bd_sym_a.redirect()},{1000,2000,300,400});
  // TTT.print_diagram();
  // TTT.print_blocks(false);

  UniTensor T3A({bd_sym_s, bd_sym_s, bd_sym_s2, bd_sym_s3});
  T3A.print_diagram();
  T3A.print_blocks(false);

  // auto tnB1 = T3A.get_block_({1,0,2,0});
  // print(tnB1);

  UniTensor T3B(
    {bd_sym_s.redirect(), bd_sym_s.redirect(), bd_sym_s2.redirect(), bd_sym_s3.redirect()});
  T3B = T3B.relabels({4, 5, 2, 6});

  T3B.print_diagram();

  T3A.contract(T3B);

  return 0;

  // UniTensor
  // T3B({bd_sym_s.redirect(),bd_sym_s.redirect(),bd_sym_s2.redirect(),bd_sym_s3.redirect()});

  // auto OutAB = T3A.contract(T3B);

  // OutAB.print_diagram();
  // OutAB.print_blocks();

  // auto T33_b = T33.relabels({"a","c","ds","r"});
  // auto Ot = T33.contract(T33_b);
  // Ot.print_diagram();
  // Ot.print_blocks(false);
  /*
  UniTensor TTT({bd_sym_a,bd_sym_a.redirect()},{1000,2000});
  UniTensor TTT2 = TTT.relabels({300,400});


  TTT.print_diagram();
  TTT2.print_diagram();

  auto TTO = TTT.contract(TTT2);
  TTO.print_diagram();
  TTO.print_blocks(false);
  */
  return 0;

  bd_sym_a.Save("ttba");

  print(bd_sym_a);

  Bond bd_r = Bond::Load("ttba.cybd");

  print(bd_r);

  return 0;

  Bond bd_sym_b = bd_sym_a.clone();

  cout << bd_sym_a << endl;

  cout << bd_sym_a.type() << endl;
  cout << bd_sym_a.dim() << endl;
  cout << bd_sym_a.Nsym() << endl;
  cout << bd_sym_a.syms() << endl;

  cout << bd_sym_a.qnums() << endl;

  print(bd_sym_a);
  print(bd_sym_b);
  bd_sym_a.combineBond_(bd_sym_b);

  print(bd_sym_a);

  exit(1);

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
