#include "Storage.hpp"
#include "Tensor.hpp"
#include "Bond.hpp"
#include <complex>

using namespace std;
using namespace tor10;

void func(boost::intrusive_ptr<Storage_base> &a){
    a = boost::intrusive_ptr<Storage_base>(new FloatStorage());
}

int main(int argc, char *argv[]){
    
    //tor10device.Print_Property();
    /*
    boost::intrusive_ptr<Storage_base> array1(new FloatStorage() );
    boost::intrusive_ptr<Storage_base> array2(new Storage_base() );
    boost::intrusive_ptr<Storage_base> array3(new Storage_base() );


    array1->Init(4);
    array1->to_(tor10device.cuda); 
    array2->Init(4);  
    array3->Init(4);
    cout << array1->dtype() << endl;
    cout << array2->dtype() << endl;
    cout << array3->dtype() << endl;

    array1->at<float>(0) = 1;
    array1->at<float>(1) = 2;
    array1->at<float>(2) = 3;
    array1->at<float>(3) = 4;

    array2 = array1->astype(tor10type.ComplexDouble);
    for(int i=0;i<4;i++)
        cout << array2->at<complex<double> >(i);
   
 
    cout << array2->dtype() << endl;    

    array2 = array2->astype(tor10type.ComplexFloat);

    // GET RAW POINTER
    float* A = array1->data<float>();
    
    // Convert type:
    //boost::intrusive_ptr<Storage_base> array3 = array2->astype(tor10type.Float);
    boost::intrusive_ptr<Storage_base> array4 = array3;

   
    cout << array3->refcount() << endl;
    array4 = array2;
    cout << array3->refcount() << endl;

    float *tt = (float*)calloc(4,sizeof(float));

    boost::intrusive_ptr<Storage_base> arrayX(new FloatStorage() );
    arrayX->_Init_byptr_safe(tt,4);
    */
    Tensor t;
    t.Init({3,4,5},tor10type.Double,tor10device.cuda); 
    Tensor v = t;
    v.at<double>({2,1,3}) = 1;
    cout << t << endl;

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

    t.to_(tor10device.cpu);
    cout << t << endl;
    Bond bd_in = Bond(3,{{0, 1,-1, 4},
                         {0, 2,-1,-4},
                         {1, 0, 2, 2}}
                       ,{Symmetry(tor10stype.Z,2),
                         Symmetry(tor10stype.Z,3),
                         Symmetry(tor10stype.U),
                         Symmetry(tor10stype.U)}
                       ,bondType::BD_KET);

    cout << bd_in << endl;
    return 0;
}
