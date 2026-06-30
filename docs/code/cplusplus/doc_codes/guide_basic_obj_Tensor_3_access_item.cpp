auto A = cytnx::arange(24).reshape(2, 3, 4);
auto B = A(0, 0, 1);
Scalar C = B.item();
double Ct = B.item<double>();

std::cout << B << std::endl;
std::cout << C << std::endl;
std::cout << Ct << std::endl;
