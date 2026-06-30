Scalar A = 10;
std::cout << A << std::endl;

auto fA = float(A);  // convert to float
std::cout << typeid(fA).name() << fA << std::endl;

// convert to complex double
auto cdA = complex128(A);
std::cout << cdA << std::endl;

// convert to complex float
auto cfA = complex64(A);
std::cout << cfA << std::endl;
