auto A = cytnx::Tensor({2, 3});
auto B = A;

std::cout << cytnx::is(B, A) << std::endl;
