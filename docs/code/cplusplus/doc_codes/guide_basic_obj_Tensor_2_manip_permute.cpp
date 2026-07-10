auto A = cytnx::arange(24).reshape(2, 3, 4);
auto B = A.permute(1, 2, 0);
std::cout << A << std::endl;
std::cout << B << std::endl;
