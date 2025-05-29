auto T = cytnx::UniTensor(cytnx::arange(9).reshape(3,3));
print(T.at({0,2}));
T.at({0,2}) = 7;
print(T.at({0,2}));
