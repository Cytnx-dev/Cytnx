// read
auto A = cytnx::Storage(10);
A.fill(10);
std::cout << A << std::endl;

A.Tofile("S1");

// load
auto B = cytnx::Storage::Fromfile("S1", cytnx::Type.Double);

std::cout << B << std::endl;
