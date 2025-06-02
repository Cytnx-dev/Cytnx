auto x = cytnx::ones(4);
auto H = cytnx::arange(16).reshape(4, 4);

auto y = cytnx::linalg::Dot(H, x);

cout << x << endl;
cout << H << endl;
cout << y << endl;
