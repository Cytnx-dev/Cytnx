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
        MyOp2(int size, int dtype):
            LinOp("mv",size,dtype,Device.cpu){ //invoke base class constructor!
            
        }

        Tensor matvec(const Tensor &in){
            auto H = arange(100).reshape(10,10);
            return linalg::Dot(H,in);
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


    auto bdi = Bond(2,bondType::BD_KET,{{1},{-1}});
    auto bdo = bdi.clone().set_type(bondType::BD_BRA);
    

    print(bdi);
    print(bdo);

    auto Ut = UniTensor({bdi,bdi,bdo,bdo},{},2);

    Ut.get_block_({2})(0) = 1;
    auto T0 = Ut.get_block_({0});
    T0(0,0) = T0(1,1) = -1;
    T0(0,1) = T0(1,0) =  1;
    
    Ut.get_block_({-2})(0) = 1;

    Ut.print_diagram();
    print(Ut);



    Ut = Ut.permute({1,3,0,2},3,false);
    cout << Ut.is_contiguous() << endl;
    
    auto tmpU = Ut.contiguous();

    print(tmpU);

    /*    
    auto Ut2 = Ut.clone();
    Ut2.set_labels({3,4,5,6});
        
    Ut2.print_diagram();
    */
    //Contract(Ut,Ut2);


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


    auto Hopo = MyOp2(vec.shape()[0],dty);
    print(Hopo.matvec(vec));

    print(linalg::Lanczos_Gnd(&Hopo));

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
