auto A = cytnx::Tensor({2, 3});
auto B = A;
auto C = A.clone();

std::cout << cytnx::is(B, A) << std::endl;
std::cout << cytnx::is(C, A) << std::endl;
