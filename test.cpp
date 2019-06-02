#include "cytnx.hpp"
#include <complex>

using namespace std;
using namespace cytnx;

void func(boost::intrusive_ptr<Storage_base> &a){
    a = boost::intrusive_ptr<Storage_base>(new FloatStorage());
}

typedef cytnx::Accessor ac;

int main(int argc, char *argv[]){

    vector<cytnx_int64> A(10);
    for(int i=0;i<10;i++){
        A[i] = rand()%500;
    }
    cout << vec_unique(A) << endl; 


    //Device.Print_Property();
    /*
    boost::intrusive_ptr<Storage_base> array1(new FloatStorage() );
    boost::intrusive_ptr<Storage_base> array2(new Storage_base() );
    boost::intrusive_ptr<Storage_base> array3(new Storage_base() );


    array1->Init(4);
    array1->to_(Device.cuda); 
    array2->Init(4);  
    array3->Init(4);
    cout << array1->dtype() << endl;
    cout << array2->dtype() << endl;
    cout << array3->dtype() << endl;

    array1->at<float>(0) = 1;
    array1->at<float>(1) = 2;
    array1->at<float>(2) = 3;
    array1->at<float>(3) = 4;

    array2 = array1->astype(Type.ComplexDouble);
    for(int i=0;i<4;i++)
        cout << array2->at<complex<double> >(i);
   
 
    cout << array2->dtype() << endl;    

    array2 = array2->astype(Type.ComplexFloat);

    // GET RAW POINTER
    float* A = array1->data<float>();
    
    // Convert type:
    //boost::intrusive_ptr<Storage_base> array3 = array2->astype(Type.Float);
    boost::intrusive_ptr<Storage_base> array4 = array3;

   
    cout << array3->refcount() << endl;
    array4 = array2;
    cout << array3->refcount() << endl;

    float *tt = (float*)calloc(4,sizeof(float));

    boost::intrusive_ptr<Storage_base> arrayX(new FloatStorage() );
    arrayX->_Init_byptr_safe(tt,4);
    */
    
    Storage s;
    s.Init(12,Type.Double,Device.cpu);
    s.set_zeros();
    s.at<double>(4) = 3;
    cout << s << endl;    
    Storage s2 = s;
    Storage s3 = s.clone();
    cout << is(s,s2) << is(s,s3) << endl;
    cout << (s==s2) << (s==s3) << endl;
    

    Tensor x = zeros({3,4,5},Type.Double,Device.cpu);
     
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

    Bond bd_in = Bond(3,BD_KET,{{0, 1,-1, 4},
                                {0, 2,-1,-4},
                                {1, 0, 2, 2}}
                              ,{Symmetry::Zn(2),
                                Symmetry::Zn(3),
                                Symmetry::U1(),
                                Symmetry::U1()});

    cout << bd_in << endl;

    Bond bd_r = Bond(10);
    cout << bd_r << endl;
    
    Bond bd_l = Bond(10,BD_KET);
    cout << bd_l << endl;

    Bond bd_dqu1 = Bond(3, BD_BRA,{{0,2},{1,2},{3,3}});
    cout << bd_dqu1 << endl;

    return 0;
}
