// read
auto A = cytnx::Storage(10);
A.fill(10);
cout << A << endl;

A.Tofile("S1");

// load
auto B = cytnx::Storage::Fromfile("S1", cytnx::Type.Double);

cout << B << endl;
