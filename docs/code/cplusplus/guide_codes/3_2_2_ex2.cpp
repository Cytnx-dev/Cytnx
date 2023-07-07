auto A = cytnx::arange(24).reshape(2,3,4);
cout << A.is_contiguous() << endl;
cout << A << endl;

A.permute_(1,0,2);
cout << A.is_contiguous() << endl;
cout << A << endl;

A.contiguous_();
cout << A.is_contiguous() << endl;
