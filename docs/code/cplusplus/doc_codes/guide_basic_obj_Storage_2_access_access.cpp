auto A = cytnx::Storage(6);
A.set_zeros();
cout << A << endl;

A.at<double>(4) = 4;
cout << A << endl;
