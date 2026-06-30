auto A = cytnx::ones({2, 2});  // on CPU
auto B = A.to(cytnx::Device.cuda + 0);
std::cout << A << std::endl;  // on CPU
std::cout << B << std::endl;  // on GPU

A.to_(cytnx::Device.cuda);
std::cout << A << std::endl;  // on GPU
