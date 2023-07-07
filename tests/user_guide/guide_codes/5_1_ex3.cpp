Scalar A = 10;
cout << A << endl;

auto fA = float(A); // convert to float
cout << typeid(fA).name() << fA << endl;

// convert to complex double
auto cdA = complex128(A);
cout << cdA << endl;

// convert to complex float
auto cfA = complex64(A);
cout << cfA << endl;
