//#include "cytnx.hpp"
#include <complex>
#include <cstdarg>
#include <functional>
#include "torcyx.hpp"

using namespace std;
using namespace cytnx;

typedef cytnx::Accessor ac;

namespace A{
    class Tyt{
        public:
            int x;
            Tyt(){};    
    };

}

namespace B{
    using A::Tyt;
}



//-------------------------------------------
/*
Tensor myfunc(const Tensor &Tin){
    // Tin should be a 4x4 tensor.
    Tensor A = arange(25).reshape({5,5});
    A += A.permute({1,0}).contiguous()+1;
    


    return linalg::Dot(A,Tin);

}

class MyOp: public LinOp{
    using LinOp::LinOp;
    public:
        Tensor ori;
    // override!
    Tensor matvec(const Tensor &Tin){
        //cout << ori << endl;
        return linalg::Dot(ori,Tin);
    }


};
*/
 
int main(int argc, char *argv[]){


    B::Tyt xx = B::Tyt();

    torch::Tensor qr = torch::arange(12).to(torch::kFloat).reshape({4,3});
    auto out = qr.div(cytnx_int64(3));
    cout << out << endl;
    

    return 0;
    /*
    auto options = ml::type_converter.Cy2Tor(Type.Float,Device.cpu);
    cout << options << endl;
    
    auto A = torch::arange(12).reshape({4,3});
    cout << A << endl;
    
    auto out = torch::svd(A.to(torch::kFloat));
    cout << std::get<0>(out) << std::get<1>(out) << std::get<2>(out) << endl;
    return 0;
    

    auto c = linalg::Qr(qr);
    cout << c;
    cout << linalg::Matmul(c[0],c[1]);

    auto dd = linalg::Qdr(qr);
    cout << dd;
    dd[1] = linalg::Matmul(linalg::Diag(dd[1]),dd[2]);
    cout << linalg::Matmul(dd[0],dd[1]);


    Tensor rar = qr(":",ac({0,0,1,1}));
    //Tensor rar = qr[{ac::all(),ac::all()}];
    cout << rar;

    cout << qr(0,0).item<double>() << endl;


    return 0;
    auto At = cytnx::zeros(24).reshape(2,3,4);
    cout << At << endl;

    At = At[{ac(0)}];

    return 0;
    vector<double> XvA(4,6);
    
    auto XA = cytnx::Storage::from_vector(XvA);
    auto XB = cytnx::Storage::from_vector(XvA,Device.cuda);

    cout << XA << endl;
    cout << XB << endl;

    return 0 ;    
    auto TNss = zeros({3,4});

    cout << TNss - std::complex<double>(0,7) << endl;
    return 0;

    //cout << testta << endl;
    auto TNs = arange(16).astype(Type.Double).reshape({4,4});

    cout << TNs;
    cout << TNs(":","-1::-1");
    //cout << TNs(ac::all(),3)/2;
    
    //cout << TNs(3);
    //cout << linalg::Svd(TNs) << endl;
    exit(1);

    //LinOp *cu = new MyOp("mv",5,Type.Double,Device.cpu);
    //cout << cu->matvec(t);

    //free(cu);




    Tensor ttA = arange(25).reshape({5,5});
    ttA += ttA.permute({1,0}).contiguous()+1;

    auto outtr = linalg::Eigh(ttA);

    ttA = linalg::Dot(linalg::Dot(outtr[1],linalg::Diag(arange(5)+1)),outtr[1].permute({1,0}).contiguous());
    cout << ttA << endl;


    //LinOp HOp("mv",5,Type.Double,Device.cpu,myfunc);
    auto t = arange(5);
    MyOp cu("mv",5,Type.Double,Device.cpu);    
    cu.ori = ttA;
    //cout << cu.matvec(t);    
    //LinOp *tpp = &cu;
    //cout << tpp->matvec(t);    
    //exit(1);
    //cout << linalg::Eigh(ttA) << endl;

    //cout << linalg::Lanczos_ER(&cu,2);

    exit(1);
    //exit(1);




    auto TNtts = arange(16);
    cout << TNtts << endl;

    TNtts.to_(Device.cuda);
    linalg::Pow_(TNtts,2);
    cout << TNtts << endl;
    exit(1);
    TNtts[{0}] = TNtts[{1}] = 9999;

    //TNtts[{ac(4)}] = 99999;


    //auto TNget = TNtts[{ac::range(0,16,2)}];
    //cout << TNget << endl;
    cout << TNtts;
    return 0;


    TNs = TNs + TNs.permute({1,0});
    
    cout << linalg::ExpH(TNs);   

    auto Tuu = UniTensor(TNs.reshape({2,2,2,2}),2);
    Tuu.print_diagram();
    auto Tuux = linalg::ExpH(Tuu);
    Tuux.print_diagram();
    Tuux.combineBonds({0,1}); 
    Tuux.combineBonds({2,3}); 
    Tuux.print_diagram();
    cout << Tuux << endl;


    auto Tori = zeros({4,4});
    auto cTori = UniTensor(Tori,2);
    cTori.at<cytnx_double>({0,0}) = 1.456;
    cout << cTori << endl;
    cout << Tori << endl;

    return 0;

    */
     
    return 0;
}
