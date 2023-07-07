auto A = cytnx::arange(12).reshape(3,4);
cout << A << endl;

auto B = cytnx::ones({3,4})*4;
cout << B << endl;

auto C = A * B;
cout << C << endl;
