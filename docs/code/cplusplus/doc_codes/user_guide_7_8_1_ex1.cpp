auto T = cytnx::UniTensor(cytnx::arange(9).reshape(3, 3));
print(T.at({0, 2}));
