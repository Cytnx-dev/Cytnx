auto x = cytnx::ones(4);
auto H = cytnx::arange(16).reshape(4, 4);

auto y = cytnx::linalg::Dot(H, x);

std::cout << x << std::endl;
std::cout << H << std::endl;
std::cout << y << std::endl;
