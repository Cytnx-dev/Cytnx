typedef ac = cytnx::Accessor;
auto A = cytnx::arange(24).reshape(2, 3, 4);
auto B = cytnx::zeros({3, 2});

// [get] this is equal to A[0,:,1:4:2] in Python:
auto C = A.get({ac(0},ac::all(),ac::range(1,4,2)
});

// [set] this is equal to A[1,:,0:4:2] = B in Python:
A.set({ac(1), ac::all(), ac::range(0, 4, 2)}, B);
