#include "cytnx.hpp"
#include <complex>
#include <cstdarg>
#include <functional>
 
#include "hptt.h"
//#include "cutt.h"
using namespace std;
using namespace cytnx;

typedef cytnx::Accessor ac;


Scalar generic_func(const Tensor &input, const std::vector<Scalar> &args){
    auto out = input+args[0];
    out=args[1]+out;
    return out(0,0).item();
}


class test{
    public:
        
        Tensor tff(const Tensor &T){
            auto A = T.reshape(2,3,4);
            return A;
        }

};


//-------------------------------------------

Tensor myfunc(const Tensor &Tin){
    // Tin should be a 4x4 tensor.
    Tensor A = arange(25).reshape({5,5});
    A += A.permute({1,0}).contiguous()+1;
    
    return linalg::Dot(A,Tin);

}

class MyOp2: public LinOp{
    public:
        
        Tensor H;
        MyOp2():
            LinOp("mv",0,Type.Double,Device.cpu){ //invoke base class constructor!
            auto T = arange(100).reshape(10,10);
            T = T + T.permute(1,0);
            print(linalg::Eigh(T));            
        }

        UniTensor matvec(const UniTensor &in){
        
            auto T = arange(100).reshape(10,10);
            T = T + T.permute(1,0);
            auto H = UniTensor(T,1);
            
            auto out = Contract(H,in);
            out.set_labels(in.labels());
            
            return out;
        }

};


#define cuttCheck(stmt) do {                                 \
  cuttResult err = stmt;                            \
  if (err != CUTT_SUCCESS) {                          \
    fprintf(stderr, "%s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
    exit(1); \
  }                                                  \
} while(0)


Scalar run_DMRG(tn_algo::MPO &mpo, tn_algo::MPS &mps, int Nsweeps, std::vector<tn_algo::MPS> ortho_mps={}, double weight=40){

    auto model = tn_algo::DMRG(mpo,mps,ortho_mps,weight);

    model.initialize();
    Scalar E;
    for(int xi=0;xi<Nsweeps; xi++){
        E = model.sweep();
        cout << "sweep " << xi << "/" << Nsweeps << " | Enr: " << E << endl;
    }
    return E;
}





int main(int argc, char *argv[]){



    auto X0 = arange(32).reshape(2,2,2,2,2);
    
    auto X1 = X0.permute(0,3,2,1,4);
    auto X2 = X0.permute(2,0,1,4,3);

    auto X1c = X1.contiguous();
    auto X2c = X2.contiguous();

    cout << X1c + X2c << endl;
    cout << X1c.is_contiguous() << X2c.is_contiguous() << endl;

    X1c+=X2c;


    X1 += X2;

    cout << X1 << endl;
    cout << X2 << endl;
    auto idd = vec_range(5);
    cout << X1._impl->invmapper() << endl;
    cout << X1._impl->mapper() << endl;

    cout << X2._impl->invmapper() << endl;
    cout << X2._impl->mapper() << endl;
    cout << vec_map(idd,X2._impl->invmapper()) << endl;

    return 0;

    print(X1._impl->mapper());
    print(X1._impl->invmapper());

    print(X2._impl->mapper());
    print(X2._impl->invmapper());


    std::vector<cytnx_uint32> _invL2R(7);

    for(int i=0;i < X1._impl->mapper().size();i++){
        _invL2R[i] = X2._impl->mapper()[X1._impl->invmapper()[i]];
    }
    print(_invL2R);


    return 0;



    return 0;

    auto Tc = zeros({4,1,1});
    Tc.Conj_();

    auto L0 = UniTensor(zeros({4,1,1}),0); //#Left boundary
    auto R0 = UniTensor(zeros({4,1,1}),0); //#Right boundary
    L0.get_block_()(0,0,0) = 1.; R0.get_block_()(3,0,0) = 1.;



    MyOp2 OP;
    auto v = UniTensor(random::normal({10},0,1,-1,99),1);

    //auto out = OP.matvec(v);

    print(linalg::Lanczos_Gnd_Ut(&OP,v));



    return 0;       

    auto ac1 = Accessor::qns({{1},{-1}});
    print(ac1);

    auto tn1 = zeros(4);
    auto tn3 = zeros(7);

    cout << algo::Concatenate(tn1,tn3);



    auto bdii = Bond(5,bondType::BD_KET,{{1},{1},{-1},{-1},{-1}});
    auto bdoo = Bond(5,bondType::BD_BRA,{{-1},{-1},{-1},{1},{1}});

    auto tTrace = UniTensor({bdii,bdii.redirect()},{},1);
    auto tTrace2 = tTrace.Dagger();

    Contract(tTrace,tTrace2);


    return 0;

    auto tit = UniTensor({bdii,bdii,bdoo,bdoo},{},2);

    for(int i=0; i < tit.get_blocks_().size(); i++){
        random::Make_normal(tit.get_block_(i),0,1);
    }

    tit.print_diagram();

    print(tit);

   


    auto outt = linalg::Svd_truncate(tit,10);


    auto outo = linalg::Svd(tit);

    //print(outt);


    print(outt[0]);
    print(outo[0]);

    return 0;


    auto bdi = Bond(2,bondType::BD_KET,{{1},{-1}});
    auto bdo = bdi.clone().set_type(bondType::BD_BRA);
    
    print(bdi);
    print(bdo);

    auto Ut = UniTensor({bdi,bdi,bdo,bdo},{},2);
    auto UtxUt = UniTensor({bdi,bdi,bdi,bdi,bdo,bdo,bdo,bdo},{},4);

    print(Ut.syms());
    auto T = Symmetry::Zn(4); 
    print(T);
    
    Ut.get_block_({2})(0) = 1;
    auto T0 = Ut.get_block_({0});
    T0(0,0) = T0(1,1) = -1;
    T0(0,1) = T0(1,0) =  1;
    
    Ut.get_block_({-2})(0) = 1;

   


    Ut.permute_({2,0,3,1}); 


    cout << "Svd" << endl;
    auto outv = linalg::Svd(Ut);

    

    auto S = outv[0];

    S.print_diagram();
    outv[1].print_diagram();
    outv[2].print_diagram();



    auto Us = Contract(S,outv[1]);

    auto UsV = Contract(Us,outv[2]);

    Us.print_diagram();
    UsV.print_diagram();
    Ut.print_diagram();


    print(Ut);
    print(UsV);

    cout << "[OK]" << endl;
    

    //Ut.print_diagram();
    //print(Ut);
    return 0;
    /*
    print(Ut.get_blocks_().size());
    print(Ut.get_blocks_qnums());
    for(int i=0;i<Ut.get_blocks_().size();i++){
        print(Ut.get_blocks_()[i].shape());
    }
    
    print(UtxUt.get_blocks_().size());
    print(UtxUt.get_blocks_qnums());    
    print(UtxUt.get_blocks_());
    for(int i=0;i<UtxUt.get_blocks_().size();i++){
        print(UtxUt.get_blocks_()[i].shape());
    }
    */

    /*
    auto outv = linalg::Svd(Ut);

    outv[0].print_diagram(true);
    outv[1].print_diagram(true);
    outv[2].print_diagram(true);
    */
    return 0;



    Ut = Ut.permute({1,0,2,3});

    cout << Ut.is_contiguous() << endl;

    cout << Ut.is_braket_form() << endl;

    return 0;


     
   

    


    return 0; 


    int Nn = 8; // # of stars:
    int chi = 8;
    vector<cytnx_uint64> phys_dims;

    for(int i=0;i< Nn;i++){
        phys_dims.push_back(16);
        phys_dims.push_back(4);
    }


    auto mps0 = tn_algo::MPS(Nn*2,phys_dims,chi);
    print(mps0);

    for(int i=0;i<mps0.size();i++){
        print(mps0.data()[i].shape());
    }

    return 0;



    return 0;


    return 0;

    auto dty = Type.Float;
    auto vec = arange(10).astype(dty);
    vec/=vec.Norm().item();


    return 0;




    Scalar scA = int(5);
    Scalar scB = 4.77;

    cout << (scA < scB) << endl;    
    return 0;

    auto Trt = arange(30).reshape(1,30);
    
    auto uTrt = UniTensor(Trt,1);

    std::cout << linalg::Svd(uTrt);

    return 0;

    std::complex<double> j = {0,1};
    cout << j << endl;
    auto Sx = physics::spin(0.5,'x');
    auto Sy = physics::spin(0.5,'y');
    auto Sp = Sx + j*Sy;
    auto Sm = Sx - j*Sy;
    cout << Sp <<endl;
    return 0;


    auto S00 = Storage(30);
    cytnx_int64 ia = 5;
    cytnx_int64 ib = 6;
    Tensor T00 = Tensor::from_storage(S00).reshape(ia,ib);

    T00 = T00.reshape(5,3,2);

    T00 = T00.reshape(30);


    return 0;

    auto Arrr = Tensor({2,3,4});
    auto Tnt = test();
    Tnt.tff(Arrr);    
    return 0;


 

     
    return 0;
}
