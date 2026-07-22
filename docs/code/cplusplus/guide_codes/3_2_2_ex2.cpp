auto A = cytnx::arange(24).reshape(2, 3, 4);
std::cout << A.is_contiguous() << std::endl;
std::cout << A << std::endl;

A.permute_(1, 0, 2);
std::cout << A.is_contiguous() << std::endl;
std::cout << A << std::endl;

A.contiguous_();
std::cout << A.is_contiguous() << std::endl;
