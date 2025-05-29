auto A = cytnx::Storage(4);

auto B = A.to(cytnx::Device.cuda);
cout << A.device_str() << endl;
cout << B.device_str() << endl;

A.to_(cytnx::Device.cuda);
cout << A.device_str() << endl;
