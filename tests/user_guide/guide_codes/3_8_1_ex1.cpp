  auto A = cytnx::zeros({3,4,5});
  auto B = A;

  cout << is(B,A) << endl;
