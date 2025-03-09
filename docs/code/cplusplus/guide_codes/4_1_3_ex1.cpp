auto A = cytnx::arange(10).reshape(2,5);
auto B = A.storage();

cout << A << endl;
cout << B << endl;
