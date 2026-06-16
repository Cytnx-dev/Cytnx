using namespace cytnx;
class MyOp : public LinOp {
 public:
  double AddConst;

  MyOp(double aconst)
      :  // invoke base class constructor!
        LinOp(4, Type.Double, Device.cpu) {
    this->AddConst = aconst;
  }
};
