auto A = cytnx::ones({3, 4, 5});
auto B = cytnx::ones({4, 5}) * 2;
cout << A << endl;
cout << B << endl;

A.append(B);
cout << A << endl;
