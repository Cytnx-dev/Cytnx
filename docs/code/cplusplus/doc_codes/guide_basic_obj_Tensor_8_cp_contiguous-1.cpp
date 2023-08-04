auto A = cytnx::zeros({2, 3, 4});
auto B = A.permute(0, 2, 1);

cout << A.is_contiguous() << endl;
cout << B.is_contiguous() << endl;
