auto A = cytnx::arange(24).reshape(2,3,4);
auto B = cytnx::zeros({3,2});
cout << A << endl;
cout << B << endl;

A(1,":","::2") = B;
cout << A << endl;

A(0,"::2",2) = 4;
cout << A << endl;
