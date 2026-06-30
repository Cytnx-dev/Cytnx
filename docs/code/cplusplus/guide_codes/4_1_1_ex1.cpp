auto A = cytnx::Storage(10);
A.set_zeros();

auto B = A.astype(cytnx::Type.ComplexDouble);

std::cout << A << std::endl;
std::cout << B << std::endl;
