auto A = cytnx::Storage(6);
std::cout << A << std::endl;

Scalar elemt = A.at(4);
std::cout << elemt << std::endl;

A.at(4) = 4;
std::cout << A << std::endl;
