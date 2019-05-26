#include "cytnx.hpp"
#include <complex>

using namespace std;
using namespace cytnx;

void func(boost::intrusive_ptr<Storage_base> &a){
    a = boost::intrusive_ptr<Storage_base>(new FloatStorage());
}

int main(int argc, char *argv[]){
    


    //cytnxdevice.Print_Property();
    /*
    boost::intrusive_ptr<Storage_base> array1(new FloatStorage() );
    boost::intrusive_ptr<Storage_base> array2(new Storage_base() );
    boost::intrusive_ptr<Storage_base> array3(new Storage_base() );


    array1->Init(4);
    array1->to_(cytnxdevice.cuda); 
    array2->Init(4);  
    array3->Init(4);
    cout << array1->dtype() << endl;
    cout << array2->dtype() << endl;
    cout << array3->dtype() << endl;

    array1->at<float>(0) = 1;
    array1->at<float>(1) = 2;
    array1->at<float>(2) = 3;
    array1->at<float>(3) = 4;

    array2 = array1->astype(cytnxtype.ComplexDouble);
    for(int i=0;i<4;i++)
        cout << array2->at<complex<double> >(i);
   
 
    cout << array2->dtype() << endl;    

    array2 = array2->astype(cytnxtype.ComplexFloat);

    // GET RAW POINTER
    float* A = array1->data<float>();
    
    // Convert type:
    //boost::intrusive_ptr<Storage_base> array3 = array2->astype(cytnxtype.Float);
    boost::intrusive_ptr<Storage_base> array4 = array3;

   
    cout << array3->refcount() << endl;
    array4 = array2;
    cout << array3->refcount() << endl;

    float *tt = (float*)calloc(4,sizeof(float));

    boost::intrusive_ptr<Storage_base> arrayX(new FloatStorage() );
    arrayX->_Init_byptr_safe(tt,4);
    */

    Storage s;
    s.Init(12,cytnxtype.Double,cytnxdevice.cpu);
    s.at<double>(4) = 3;
    cout << s << endl;    

    Tensor x({3,4,5},cytnxtype.Double,cytnxdevice.cpu);
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

    Tensor a({2,3},cytnxtype.Double,cytnxdevice.cpu);
    a.at<double>({0,0}) = 3; a.at<double>({0,1}) = 2; a.at<double>({0,2}) = 2;
    a.at<double>({1,0}) = 2; a.at<double>({1,1}) = 3; a.at<double>({1,2}) = -2;

    vector<Tensor> out = linalg::Svd(a,false,false);
    cout << out[0] ;
    return 0;
/*
    //Tensor t;
    //t.Init({3,4,5},cytnxtype.Double,cytnxdevice.cpu); 
    Tensor t({3,4,5},cytnxtype.Double,cytnxdevice.cpu);
    Tensor v = t;
    v.at<double>({2,1,3}) = 1;
    cout << t << endl;
    return 0;
    t.permute_({1,0,2});
    cout << t << endl;
    cout << t.is_contiguous() << endl;
    
    //t.permute_({1,0,2});
    //cout << t << endl;
    //cout << t.is_contiguous() << endl;

    t.Contiguous_();
    cout << t<< endl;
    cout << t.is_contiguous() << endl;
    t.Reshape_({2,3,2,5});
    cout << t << endl;
    cout << t.shape() << endl;

    t.to_(cytnxdevice.cpu);
    cout << t << endl;
    Bond bd_in = Bond(3,{{0, 1,-1, 4},
                         {0, 2,-1,-4},
                         {1, 0, 2, 2}}
                       ,{Symmetry(cytnxstype.Z,2),
                         Symmetry(cytnxstype.Z,3),
                         Symmetry(cytnxstype.U),
                         Symmetry(cytnxstype.U)}
                       ,bondType::BD_KET);

    cout << bd_in << endl;
    */
    return 0;
}
