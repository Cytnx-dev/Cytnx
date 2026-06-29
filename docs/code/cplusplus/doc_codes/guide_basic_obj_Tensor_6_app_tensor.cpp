auto A = cytnx::ones({3, 4, 5});
auto B = cytnx::ones({4, 5}) * 2;
std::cout << A << std::endl;
std::cout << B << std::endl;

A.append(B);
std::cout << A << std::endl;
