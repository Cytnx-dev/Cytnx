auto A = cytnx::arange(24).reshape(2, 3, 4);
std::cout << A << std::endl;

auto B = A(0, ":", "1:4:2");
std::cout << B << std::endl;

auto C = A(":", 1);
std::cout << C << std::endl;
