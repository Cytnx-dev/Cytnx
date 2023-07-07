auto A = cytnx::Tensor({2,3});
auto B = A;
auto C = A.clone();

cout << cytnx::is(B,A) << endl;
cout << cytnx::is(C,A) << endl;
