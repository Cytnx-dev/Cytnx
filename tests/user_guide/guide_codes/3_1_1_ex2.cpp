auto A = cytnx::arange(10);     //rank-1 Tensor from [0,10) with step 1
auto B = cytnx::arange(0,10,2); //rank-1 Tensor from [0,10) with step 2
auto C = cytnx::ones({3,4,5});  //Tensor of shape (3,4,5) with all elements set to one.
auto D = cytnx::eye(3);          //Tensor of shape (3,3) with diagonal elements set to one, all other entries are zero.
