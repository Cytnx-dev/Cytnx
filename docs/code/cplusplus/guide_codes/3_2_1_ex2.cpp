auto A = cytnx::arange(24);
std::cout << A << std::endl;
A.reshape_(2, 3, 4);
std::cout << A << std::endl;
