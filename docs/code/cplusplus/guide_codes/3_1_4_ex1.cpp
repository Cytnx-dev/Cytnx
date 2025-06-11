auto A = cytnx::ones({3, 4}, cytnx::Type.Int64);
auto B = A.astype(cytnx::Type.Double);
cout << A.dtype_str() << endl;
cout << B.dtype_str() << endl;
