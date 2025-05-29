auto A = cytnx::random::normal(
  {3, 4, 5}, 0., 1.);  // Tensor of shape (3,4,5) with all elements distributed according
                       // to a normal distribution around 0 with standard deviation 1
auto B = cytnx::random::uniform({3, 4, 5}, -1., 1.);  // Tensor of shape (3,4,5) with all elements
                                                      // distributed uniformly between -1 and 1
