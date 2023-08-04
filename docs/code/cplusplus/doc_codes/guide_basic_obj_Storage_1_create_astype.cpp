auto A = cytnx::Storage(10);
A.set_zeros();

auto B = A.astype(cytnx::Type.ComplexDouble);

cout << A << endl;
cout << B << endl;
