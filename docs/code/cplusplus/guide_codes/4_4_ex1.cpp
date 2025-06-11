vector<double> vA(4, 6);

auto A = cytnx::Storage::from_vector(vA);
auto B = cytnx::Storage::from_vector(vA, cytnx::Device.cuda);

cout << A << endl;
cout << B << endl;
