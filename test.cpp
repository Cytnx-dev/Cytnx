#include "cytnx.hpp"
#include <complex>
using namespace std;
using namespace cytnx;
using namespace cytnx_extension;

namespace cyx = cytnx_extension;
typedef cytnx::Accessor ac;

int main(int argc, char *argv[]){


    




    /*

    Hn.print_diagram();
    Hn.Transpose_();
    Hn.print_diagram();
    
    //return 0;
    auto Bd1 = cyx::Bond(2,cyx::BD_KET); //# 1 = 0.5 , so it is spin-1/2
    auto Bd2 = cyx::Bond(2,cyx::BD_BRA);//# 1 = 0.5, so it is spin-1/2
    auto Htag = cyx::CyTensor({Bd1,Bd1,Bd2,Bd2},{},1);
    Htag.print_diagram();
    Htag.Transpose_();
    Htag.print_diagram();
    */


    //return 0;
    // Ket = IN
    // Bra = OUT
    auto Bd_i = cyx::Bond(3,cyx::BD_KET,{{2},{0},{-2}},{cyx::Symmetry::U1()}); //# 1 = 0.5 , so it is spin-1
    auto Bd_o = cyx::Bond(3,cyx::BD_BRA,{{2},{0},{-2}},{cyx::Symmetry::U1()});//# 1 = 0.5, so it is spin-1
    auto H = cyx::CyTensor({Bd_i,Bd_i,Bd_o,Bd_o},{},2);
    

    H.set_elem<cytnx_double>({0,0,0,0},1);
    auto HHH = H.Transpose();

    cout << HHH << endl;

    //cout << H ;
    //cout << H.getTotalQnums();
    //cout << H.get_elem<cytnx_double>({0,0,0,0});

    //cout << H.get_block({4});
    
    //H.print_diagram();

    //H.permute_({0,3,1,2});   
    //H.print_diagram();    

    return 0;
    auto HT = linalg::Kron(physics::spin(1,'z'),physics::spin(1,'z')) + 
              linalg::Kron(physics::spin(1,'x'),physics::spin(1,'x')) +
              linalg::Kron(physics::spin(1,'y'),physics::spin(1,'y'));
    cout << HT.real() << endl;
    HT = HT.real().reshape({3,3,3,3});
    //HT.permute_({0,3,1,2});
    
    for(int i=0;i< HT.shape()[0];i++)
        for(int j=0;j<HT.shape()[1];j++)
            for(int k=0;k< HT.shape()[2];k++)
                for(int l=0;l<HT.shape()[3];l++){
                    if(abs(HT.at<cytnx_double>({i,j,k,l}))>1.0e-15)
                        H.at<cytnx_double>({i,j,k,l}) = HT.at<cytnx_double>({i,j,k,l});
                }
    //HT.reshape_({3,27});
    HT.permute_({1,2,3,0});
    HT.reshape_({3,-1});    
    cout << linalg::Matmul(HT,HT.permute({1,0}).contiguous());




    H.set_Rowrank(1 ); H.contiguous_();
    H.print_diagram();

    auto Hcp = H.Transpose();
    Hcp.contiguous_();    

    

    for(int i=0;i<Hcp.get_blocks().size();i++){
        //cout << H.get_block(i) ;
        //cout << Hcp.get_block(i) ;
        cout << linalg::Matmul(H.get_block(i),Hcp.get_block(i));
        cout << "==================\n";
    }

    
    return 0;
    
    
    
    return 0;
    
    H.get_block_({4}).at<cytnx_double>({0,0}) = HT.at<cytnx_double>({0,0});

    H.get_block_({2}).at<cytnx_double>({0,0}) = HT.at<cytnx_double>({1,1});
    H.get_block_({2}).at<cytnx_double>({0,1}) = HT.at<cytnx_double>({1,3});
    H.get_block_({2}).at<cytnx_double>({1,0}) = HT.at<cytnx_double>({3,1});
    H.get_block_({2}).at<cytnx_double>({1,1}) = HT.at<cytnx_double>({3,3});

    H.get_block_({0}).at<cytnx_double>({0,0}) = HT.at<cytnx_double>({2,2});
    H.get_block_({0}).at<cytnx_double>({0,1}) = HT.at<cytnx_double>({2,4});
    H.get_block_({0}).at<cytnx_double>({0,2}) = HT.at<cytnx_double>({2,6});
    H.get_block_({0}).at<cytnx_double>({1,0}) = HT.at<cytnx_double>({4,2});
    H.get_block_({0}).at<cytnx_double>({1,1}) = HT.at<cytnx_double>({4,4});
    H.get_block_({0}).at<cytnx_double>({1,2}) = HT.at<cytnx_double>({4,6});
    H.get_block_({0}).at<cytnx_double>({2,0}) = HT.at<cytnx_double>({6,2});
    H.get_block_({0}).at<cytnx_double>({2,1}) = HT.at<cytnx_double>({6,4});
    H.get_block_({0}).at<cytnx_double>({2,2}) = HT.at<cytnx_double>({6,6});
    cout << H << endl;
    
    return 0;
    

    auto a1 = cyx::CyTensor(cytnx::zeros({2,2,2,2}),0); a1.set_name("a1");
    auto a2 = cyx::CyTensor(cytnx::zeros({2,2,2,2}),0); a2.set_name("a2");
    auto b1 = cyx::CyTensor(cytnx::zeros({2,2,2,2}),0); b1.set_name("b1");
    auto b2 = cyx::CyTensor(cytnx::zeros({2,2,2,2}),0); b2.set_name("b2");

    auto lx1 = cyx::CyTensor(cytnx::zeros({2,2}),0); lx1.set_name("lx1"); 
    auto lx2 = cyx::CyTensor(cytnx::zeros({2,2}),0); lx2.set_name("lx2");
    auto lya1 = cyx::CyTensor(cytnx::zeros({2,2}),0); lya1.set_name("lya1"); 
    auto lya2 = cyx::CyTensor(cytnx::zeros({2,2}),0); lya2.set_name("lya2"); 
    auto lyb1 = cyx::CyTensor(cytnx::zeros({2,2}),0); lyb1.set_name("lyb1"); 
    auto lyb2 = cyx::CyTensor(cytnx::zeros({2,2}),0); lyb2.set_name("lyb2"); 
    auto lza1 = cyx::CyTensor(cytnx::zeros({2,2}),0); lza1.set_name("lza1"); 
    auto lza2 = cyx::CyTensor(cytnx::zeros({2,2}),0); lza2.set_name("lza2"); 
    auto lzb1 = cyx::CyTensor(cytnx::zeros({2,2}),0); lzb1.set_name("lzb1"); 
    auto lzb2 = cyx::CyTensor(cytnx::zeros({2,2}),0); lzb2.set_name("lzb2"); 

    auto N = cyx::Network("f.net");
    //#N.Diagram()
    N.PutCyTensor("a1",a1,true);
    N.PutCyTensor("a2",a2,true);
    N.PutCyTensor("b1",b1,true);
    N.PutCyTensor("b2",b2,true);

    N.PutCyTensor("lx1",lx1,true);
    N.PutCyTensor("lx2",lx2,true);

    N.PutCyTensor("lya1",lya1,true);
    N.PutCyTensor("lya2",lya2,true);
    N.PutCyTensor("lyb1",lyb1,true);
    N.PutCyTensor("lyb2",lyb2,true);

    N.PutCyTensor("lza1",lza1,true);
    N.PutCyTensor("lza2",lza2,true);
    N.PutCyTensor("lzb1",lzb1,true);
    N.PutCyTensor("lzb2",lzb2,true);
    cout << N;

    auto T = N.Launch(true);
    cout << N << endl;
    exit(1);

    //Tensor SZ = cytnx::physics::spin(0.5,'z');
    //cout << SZ << endl;
   
    cytnx_int64 tsc = -10;
    cout << tsc%3 << endl;
    //exit(1);

    Tensor qr = arange(12).reshape({3,4});
    cout << qr;

    Tensor qrflip = qr.get({ac::all(),ac::range(3,-1,-1)});
    cout << qrflip;

    exit(1);
    auto c = linalg::QR(qr);
    cout << c;
    c[0] = c[0][{ac::all(),ac::range(0,3)}];
    cout << linalg::Matmul(c[0],c[1]);
    exit(1);
 


    Tensor a = arange(10);
    a = a.astype(Type.ComplexFloat).to(Device.cuda+0);
    a += 1;
    cout << a << endl;

    Tensor b = linalg::Vectordot(a,a);
    cout << b << endl;
    exit(1);


    cyx::Network N2 = cyx::Network("t2.net");
    cout << N2 << endl;

    exit(1);
    Tensor ttr({3,4,5});
    random::Make_normal(ttr,0,0.1,99);
    cout << ttr ;
    
    cytnx_extension::CyTensor TTT(ttr,1);
    
    cytnx_extension::CyTensor cTT = TTT.clone();
    cytnx_extension::CyTensor cTTr = TTT;


   cout << is(cTTr,TTT) << endl;
   cout << is(cTT,TTT) << endl;

    //Device.cudaDeviceSynchronize();
    //Device.Print_Property();
/*    
    return 0;

    Tensor ttr({3,4,5});
    random::Make_normal(ttr,0,0.1,99);
    cout << ttr ;
    
    cytnx_extension::CyTensor TTT(ttr,1);
    cout << TTT;

    TTT*=3;
    cout << TTT;    

    ttr.reshape_({-1});
    cout << ttr ;

    cout << linalg::Norm(ttr);
    
    ttr.to_(Device.cuda+0);
    cout << ttr ;
    cout << linalg::Norm(ttr);


    return 0;





    Tensor ttt = arange(16).reshape({4,4});
    cout << cytnx::linalg::ExpH(ttt);
    exit(1);


    Tensor Tproto = arange(24).reshape({3,4,2});
    CyTensor U_test_svd; U_test_svd.Init(Tproto,1);
    cout << U_test_svd;


    vector<CyTensor> outCy = xlinalg::Svd(U_test_svd);
    //cout << outCy[1] << endl;
    cout << outCy << endl;

    return 0;

       
    Bond bd_dqu1 = Bond(4, BD_BRA,{{0,2},{2,0},{1,-1},{-1,1}});
    Bond bd_dqu2 = Bond(3, BD_BRA,{{1,1},{-1,-1},{0,0}});
    Bond bd_dqu3 = bd_dqu2.clone();
    bd_dqu3.set_type(BD_KET);
    Bond bd_dqu4 = bd_dqu1.clone();
    bd_dqu4.set_type(BD_KET);
    cout << bd_dqu1 << endl;
   cout << bd_dqu2 << endl;
    cout << bd_dqu3 << endl; 
    //cout << bd_dqu4.getDegeneracy({3,4}) << endl;
    //vector<vector<cytnx_int64> > comm24 = vec2d_intersect(bd_dqu2.qnums(),bd_dqu4.qnums(),false,false);
    //for(cytnx_uint64 i=0;i<comm24.size();i++){
    //    cout << comm24[i] << endl;
    //} 
    std::vector<Bond> dbds = {bd_dqu3,bd_dqu4,bd_dqu1,bd_dqu2}; 

    cout << "[OK]" << endl;        
    CyTensor dut1(dbds,{},2);
    dut1.print_diagram(true);
    dut1.permute_({2,3,0,1},1);
    dut1.print_diagram(true);

    cout << dut1.is_contiguous() << endl;

    CyTensor dut2 = dut1.contiguous();


    dut2.print_diagram(true);
    cout << dut2.is_contiguous() << endl;

    cout << dut2.get_blocks() << endl;
    //cout << dut2.get_blocks_() << endl;
    dut2.get_blocks_()[0].at<double>({0,1}) = 100;
    dut2.get_blocks_()[1].at<double>({0,1}) = 200;

    cout << dut2.get_blocks_() << endl;
    dut2.to_(Device.cuda+0);
    cout << dut2 << endl;
    //auto bcbs = dut1.getTotalQnums();
    //cout << bcbs[0] << endl;
    //cout << bcbs[1] << endl;

    return 0; 
    
    Tensor Ta({3,4,2},Type.Double);
    cout << Ta.dtype_str() << endl;
    cout << Ta.device_str() << endl;
    cout <<  Ta << endl;
    Ta.permute_({0,2,1});
    cout << Ta << endl;
    Tensor Tc = Ta.contiguous();
    cout << Tc << endl;


    Storage S1(10,Type.ComplexDouble,Device.cuda+0);
    for(unsigned int i=0;i<S1.size();i++){
        S1.at<cytnx_complex128>(i) = cytnx_complex128(i,i+1);
    }
    cout << S1 << endl;
    
    Storage S1r = S1.real();
    cout << S1r << endl;    

    Storage S1i = S1.imag();
    cout << S1i << endl;

    Storage S2 = S1.astype(Type.ComplexFloat);
    Storage S2r = S2.real();
    Storage S2i = S2.imag();
    
    cout << S2 << S2r << S2i << endl;

    return 0;

    Tensor xA = arange(12).astype(Type.Double).reshape({3,4});
    Tensor xB = arange(12).astype(Type.Double).reshape({3,4});
    Tensor xC = linalg::Tensordot(xA,xB,{0,1},{0,1});
    cout << xA << endl;
    cout << xB << endl;
    cout << xC << endl;
    return 0;
    cout << xA ;
    for(int i=0;i<4;i++){
        for(int j=0;j<3;j++){
            
            cout << xA.at<cytnx_double>({j,i});
        }
    }
    return 0;


    //Device.Print_Property();
    cytnx_complex128 testC(1,1);
    cytnx_complex64 testCf(1,1);
    cout << pow(testCf,2) << endl;
    cout << testC << endl;
    cout << pow(testC,2) << endl;

    Tensor Dtst = arange(60).reshape({4,3,5});
    Tensor Dtst_s = Dtst[{ac(1),ac::all(),ac::range(1,3)}];
    cout << Dtst << Dtst_s ;
    Dtst = arange(10);
    Dtst.append(44);
    cout << Dtst << endl;
    return 0;



    Storage s1 ;
    s1.Init(0,Type.Double,Device.cpu);
    s1.set_zeros();
    for(int i=0;i<100;i++){
        s1.append(i);
    } 
    cout << s1 << endl;
    cout << s1.capacity() << endl;


    Storage svt;
    vector<cytnx_double> tmpVVd(30);
    for(int i=0;i<tmpVVd.size();i++){
        cout << tmpVVd[i] << endl;
    }
    svt.from_vector(tmpVVd);
    cout << svt << endl;
    

    Tensor DD1 = arange(1.,5.,1.);
    cout << DD1 << endl;

    DD1.Save("test");
    Tensor DD2;
    DD2.Load("test.cytn");
    cout << DD2 << endl;



    Tensor DD = arange(1.,5.,1.);
    Tensor sDD = arange(0.4,0.7,0.1);
    cout << DD << sDD << endl;

    cout << linalg::Tridiag(DD,sDD,true);

    Tensor Trr({1,1,1},Type.Double);
    Trr.reshape_({1,1,1,1,-1});
    cout << Trr << endl;


    Tensor Xt1 = arange(4,Type.Double).reshape({2,2});
    Tensor Xt2 = arange(12,Type.Double).reshape({4,3});
    cout << Xt1 << endl;
    cout << Xt2 << endl;
    cout << linalg::Kron(Xt1,Xt2);
    
    return 0;

    
    Tensor A = arange(10,Type.Double);
    Tensor B = arange(0,1,0.1,Type.Double);
    cout << A << B << endl;
    Tensor C = linalg::Vectordot(A,B);
    cout << C.item<double>();

    A = zeros(4); A.at<double>({1}) = 1; A.at<double>({2}) = 1;
    B = zeros(4); B.at<double>({0}) = 1; B.at<double>({3}) = 1;
    
    A.reshape_({2,2});
    B.reshape_({2,2});
    cout << linalg::Kron(B,B) << endl;
    
    return 0;
    //return 0;
    
           
    Storage s;
    s.Init(12,Type.Double,Device.cpu);
    s.set_zeros();
    s.at<double>(4) = 3;
    cout << s << endl;    
    Storage s2 = s;
    Storage s3 = s.clone();
    cout << is(s,s2) << is(s,s3) << endl;
    cout << (s==s2) << (s==s3) << endl;
    
    Tensor x;
    x  = zeros({3,4,5},Type.Double,Device.cpu);
    //Tensor x = zeros({3,4,5});
    cout << x << endl; 
    x.fill(5.);
    cout << x << endl;
    Tensor b = x.clone();
    Tensor c = linalg::Add(1,x);
    Tensor d = c + c;
    Tensor e = d + 3;
    Tensor f = 3 + d;
    f += 3;
    f += 3.;
    //f+=3;
    //f+=f;
    cout << f << endl;
    
    Tensor g = c - c;
    Tensor h = g - 3.;
    Tensor i = 3. - g;
    //i-=3.;
    //i-=i;
    cout << i << endl;
    
    Tensor y = i.get({ac::all(),ac(2),ac::all()});
    cout << y << endl;

    Tensor a = zeros({2,3},Type.Double,Device.cpu);
    //Tensor a = zeros({2,3});
    a.at<double>({0,0}) = 3; a.at<double>({0,1}) = 2; a.at<double>({0,2}) = 2;
    a.at<double>({1,0}) = 2; a.at<double>({1,1}) = 3; a.at<double>({1,2}) = -2;

    vector<Tensor> out = linalg::Svd(a,false,false);
    cout << out[0] ;
    
    
    Tensor Zo = zeros(10,Type.Double,Device.cpu);
    Tensor Zo2 = zeros({3,4});
    Tensor Zp = arange(10);
    Tensor Zc = arange(0.1,0,-0.2,Type.ComplexDouble);
    cout << Zc << endl;
    
    Zp.reshape_({2,5});
    cout << Zp << endl;
    Tensor tmp = Zp.get({ac::all(),ac::range(0,2)});
    cout << tmp;
    Zp.set({ac::all(),ac::range(1,3)}, tmp);
    cout << Zp;
    Zp.set({ac::all(),ac::range(1,3)}, 4);
    cout << Zp;
*/

/*
    Bond bd_in = Bond(3,BD_KET,{{0,1,-1, 4},
                                {0,2,-1,-4},
                                {1,0, 2, 2}}
                              ,{Symmetry::Zn(2),
                                Symmetry::Zn(3),
                                Symmetry::U1(),
                                Symmetry::U1()});

    cout << bd_in << endl;


     
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
    //x1.reshape_({2,2,3,2});
    //cout << x1 << endl;

    //Tensor tmp = zeros({3,2,4});
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
