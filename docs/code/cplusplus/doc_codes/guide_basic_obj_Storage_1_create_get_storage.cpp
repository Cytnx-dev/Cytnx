auto A = cytnx::arange(10).reshape(2, 5);
auto B = A.storage();

std::cout << A << std::endl;
std::cout << B << std::endl;
