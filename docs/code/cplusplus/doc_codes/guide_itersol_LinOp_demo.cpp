auto myop = MyOp(7);
auto x = cytnx::arange(4);
auto y = myop.matvec(x);

cout << x << endl;
cout << y << endl;
