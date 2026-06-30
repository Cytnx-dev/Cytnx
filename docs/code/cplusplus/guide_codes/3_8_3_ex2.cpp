auto C = B.contiguous();

std::cout << C << std::endl;
std::cout << C.is_contiguous() << std::endl;

std::cout << C.same_data(B) << std::endl;
