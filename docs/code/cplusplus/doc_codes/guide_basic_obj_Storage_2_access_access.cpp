auto A = cytnx::Storage(6);
A.set_zeros();
std::cout << A << std::endl;

A.at<double>(4) = 4;
std::cout << A << std::endl;
