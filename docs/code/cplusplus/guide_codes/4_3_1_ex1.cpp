auto A = cytnx::Storage(4);
A.set_zeros();
std::cout << A << std::endl;

A.append(500);
std::cout << A << std::endl;
