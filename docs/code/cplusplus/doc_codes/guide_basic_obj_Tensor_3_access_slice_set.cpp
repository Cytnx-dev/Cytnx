auto A = cytnx::arange(24).reshape(2, 3, 4);
auto B = cytnx::zeros({3, 2});
std::cout << A << std::endl;
std::cout << B << std::endl;

A(1, ":", "::2") = B;
std::cout << A << std::endl;

A(0, "::2", 2) = 4;
std::cout << A << std::endl;
