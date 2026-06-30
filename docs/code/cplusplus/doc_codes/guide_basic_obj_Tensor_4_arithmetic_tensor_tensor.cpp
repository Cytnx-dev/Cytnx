auto A = cytnx::arange(12).reshape(3, 4);
std::cout << A << std::endl;

auto B = cytnx::ones({3, 4}) * 4;
std::cout << B << std::endl;

auto C = A * B;
std::cout << C << std::endl;
