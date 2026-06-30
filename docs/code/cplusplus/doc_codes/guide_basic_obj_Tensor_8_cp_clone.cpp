auto A = cytnx::zeros({3, 4, 5});
auto B = A.clone();

std::cout << is(B, A) << std::endl;
