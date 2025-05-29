auto C = B.contiguous();

cout << C << endl;
cout << C.is_contiguous() << endl;

cout << C.same_data(B) << endl;
