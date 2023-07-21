auto A = cytnx::zeros({2, 3, 4});
auto B = A.permute(0, 2, 1);

cout << A << endl;
cout << B << endl;

cout << is(B, A) << endl;
