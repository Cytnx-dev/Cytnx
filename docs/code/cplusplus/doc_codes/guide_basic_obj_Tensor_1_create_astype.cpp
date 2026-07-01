auto A = cytnx::ones({3, 4}, cytnx::Type.Int64);
auto B = A.astype(cytnx::Type.Double);
std::cout << A.dtype_str() << std::endl;
std::cout << B.dtype_str() << std::endl;
