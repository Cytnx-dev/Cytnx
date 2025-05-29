using namespace cytnx;
class MyOp : public LinOp {
 public:
  MyOp() : LinOp("mv", 6) {
    // Create Hermitian Matrix
    A = arange(36).reshape(6, 6);
    A += A.permute(1, 0);
  }

 private:
  Tensor A;
  Tensor matvec(const Tensor &v) override { return linalg::Dot(A, v); }
};

auto op = MyOp();

auto v0 = arange(6);  // trial state
auto ev = linalg::Lanczos_ER(&op, 1, true, 10000, 1.0e-14, false, v0);

cout << ev[0] << endl;  // eigenval
cout << ev[1] << endl;  // eigenvec
