auto A = cytnx::arange(24).reshape(2,3,4);
auto B = A(0,0,1);
Scalar C = B.item();
double Ct = B.item<double>();

cout << B << endl;
cout << C << endl;
cout << Ct << endl;
