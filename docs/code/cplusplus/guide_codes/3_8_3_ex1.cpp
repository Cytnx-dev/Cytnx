auto A = cytnx::zeros({2, 3, 4});
auto B = A.permute(0, 2, 1);

std::cout << A.is_contiguous() << std::endl;
std::cout << B.is_contiguous() << std::endl;
