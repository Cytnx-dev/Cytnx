auto A = cytnx::ones({3, 4});
std::cout << A << std::endl;

auto B = A + 4;
std::cout << B << std::endl;

auto C = A - std::complex<double>(0, 7);  // type promotion
std::cout << C << std::endl;
