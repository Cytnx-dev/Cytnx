auto myop = MyOp(7);
auto x = cytnx::arange(4);
auto y = myop.matvec(x);

std::cout << x << std::endl;
std::cout << y << std::endl;
