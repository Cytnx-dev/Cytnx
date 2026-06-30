auto A = cytnx::Storage(4);

auto B = A.to(cytnx::Device.cuda);
std::cout << A.device_str() << std::endl;
std::cout << B.device_str() << std::endl;

A.to_(cytnx::Device.cuda);
std::cout << A.device_str() << std::endl;
