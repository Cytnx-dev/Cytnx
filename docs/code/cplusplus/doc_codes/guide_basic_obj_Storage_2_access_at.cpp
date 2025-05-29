auto A = cytnx::Storage(6);
cout << A << endl;

Scalar elemt = A.at(4);
cout << elemt << endl;

A.at(4) = 4;
cout << A << endl;
