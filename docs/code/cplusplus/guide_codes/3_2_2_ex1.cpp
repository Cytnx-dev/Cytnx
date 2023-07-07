auto A = cytnx::arange(24).reshape(2,3,4);
auto B = A.permute(1,2,0);
cout << A << endl;
cout << B << endl;
