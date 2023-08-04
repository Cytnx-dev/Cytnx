using namespace cytnx;
class MyOp : public LinOp {
 public:
  double AddConst;

  MyOp(double aconst) : LinOp("mv", 4, Type.Double, Device.cpu) {  // invoke base class constructor!
    this->AddConst = aconst;
  }

  Tensor matvec(const Tensor& v) override {
    auto out = v.clone();
    out(0) = v(3);  // swap
    out(3) = v(0);  // swap
    out(1) += this->AddConst;  // add const
    out(2) += this->AddConst;  // add const
    return out;
  }
};
auto myop = MyOp(7);
auto x = cytnx::arange(4);
auto y = myop.matvec(x);

cout << x << endl;
cout << y << endl;
