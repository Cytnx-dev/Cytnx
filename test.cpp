#include "cytnx.hpp"
#include <complex>


using namespace std;
using namespace cytnx;


typedef cytnx::Accessor ac;

int main(int argc, char *argv[]){
    /*
    vector<cytnx_int64> A(10);
    for(int i=0;i<10;i++){
        A[i] = rand()%500;
    }
    cout << vec_unique(A) << endl; 
    */

    //Device.Print_Property();
    
           
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

    Bond bd_in = Bond(3,BD_KET,{{0,0,1},
                                {1,2,0},
                                {-1,-1,2},
                                {4,-4,2}}
                              ,{Symmetry::Zn(2),
                                Symmetry::Zn(3),
                                Symmetry::U1(),
                                Symmetry::U1()});

    cout << bd_in << endl;

    Bond bd_r = Bond(10);
    cout << bd_r << endl;
    
    Bond bd_l = Bond(10,BD_KET);
    cout << bd_l << endl;

    Bond bd_dqu1 = Bond(3, BD_BRA,{{0,2,3},{1,2,4}});
    Bond bd_dqu2 = Bond(5, BD_BRA,{{0,2,3,-2,-1},{1,2,4,-4,-2}});
    Bond bd_dqu3 = bd_dqu1.combineBond(bd_dqu2);
    cout << bd_dqu1 << endl;
   cout << bd_dqu2 << endl;
    cout << bd_dqu3 << endl; 
     
    Bond bd_1 = Bond(3);
    Bond bd_2 = Bond(5);
    Bond bd_3 = Bond(4);
    Bond bd_4 = bd_1.combineBond(bd_2);
    //std::cout << bd_4 << std::endl;

    std::vector<Bond> bds = {bd_1,bd_2,bd_3}; 
    std::vector<cytnx_int64> labels = {};
        
    UniTensor ut1(bds,{},2);
    UniTensor ut2 = ut1.clone();
    ut1.print_diagram();
    cout << ut1 << endl;
    ut1.combineBonds({2,0},true,false);
    ut1.print_diagram();
    ut2.print_diagram();
    ut2.set_label(2,-4);
    ut2.print_diagram();

    UniTensor ut3 = Contract(ut1,ut2);
    ut3.print_diagram();




    //cout << ut1 << endl;

    //cout << ut2 << endl;


    /*
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
       
    UniTensor re(sss,2); // construct by block will not copy, and share same memory.
    cout << re << endl;
    */
     
    return 0;
}
