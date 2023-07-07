auto A = cytnx::ones({3,4});
auto B = cytnx::arange(12).reshape(3,4);

// these two are equivalent to C = A+B;
auto C = A.Add(B);
auto D = cytnx::linalg::Add(A,B);

// this is equivalent to A+=B;
A.Add_(B);
