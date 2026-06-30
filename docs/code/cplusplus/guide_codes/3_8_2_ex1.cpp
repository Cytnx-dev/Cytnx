auto A = cytnx::zeros({2, 3, 4});
auto B = A.permute(0, 2, 1);

std::cout << A << std::endl;
std::cout << B << std::endl;

std::cout << is(B, A) << std::endl;
