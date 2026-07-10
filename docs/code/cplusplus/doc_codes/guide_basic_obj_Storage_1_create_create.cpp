auto A = cytnx::Storage(10, cytnx::Type.Double, cytnx::Device.cpu);
A.set_zeros();

std::cout << A << std::endl;
