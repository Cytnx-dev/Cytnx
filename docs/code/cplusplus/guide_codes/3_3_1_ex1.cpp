auto A = cytnx::arange(24).reshape(2,3,4);
cout << A << endl;

auto B = A(0,":","1:4:2");
cout << B << endl;

auto C = A(":",1);
cout << C << endl;
