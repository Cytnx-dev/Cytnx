auto A = cytnx::ones({2,2}); //on CPU
auto B = A.to(cytnx::Device.cuda+0);
cout << A << endl; // on CPU
cout << B << endl; // on GPU

A.to_(cytnx::Device.cuda);
cout << A << endl; // on GPU
