auto A = cytnx::Tensor({2,3});
auto B = A;

cout << cytnx::is(B,A) << endl;
