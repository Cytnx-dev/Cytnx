auto A = cytnx::ones({3, 4});
cout << A << endl;

auto B = A + 4;
cout << B << endl;

auto C = A - std::complex<double>(0, 7);  // type promotion
cout << C << endl;
