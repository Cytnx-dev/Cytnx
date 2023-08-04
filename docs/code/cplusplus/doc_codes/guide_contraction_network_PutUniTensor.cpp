// initialize tensors
auto w = cytnx::UniTensor(cytnx::random::normal({2, 2, 2, 2}, 0., 1.));
auto c0 = cytnx::UniTensor(cytnx::random::normal({8, 8}, 0., 1.));
auto c1 = cytnx::UniTensor(cytnx::random::normal({8, 8}, 0., 1.));
auto c2 = cytnx::UniTensor(cytnx::random::normal({8, 8}, 0., 1.));
auto c3 = cytnx::UniTensor(cytnx::random::normal({8, 8}, 0., 1.));
auto t0 = cytnx::UniTensor(cytnx::random::normal({8, 2, 8}, 0., 1.));
auto t1 = cytnx::UniTensor(cytnx::random::normal({8, 2, 8}, 0., 1.));
auto t2 = cytnx::UniTensor(cytnx::random::normal({8, 2, 8}, 0., 1.));
auto t3 = cytnx::UniTensor(cytnx::random::normal({8, 2, 8}, 0., 1.));

// initialize network object from ctm.net file
Network net = cytnx::Network("ctm.net");

// put tensors
net.PutUniTensor("w", w);
net.PutUniTensor("c0", c0);
net.PutUniTensor("c1", c1);
net.PutUniTensor("c2", c2);
net.PutUniTensor("c3", c3);
net.PutUniTensor("t0", t0);
net.PutUniTensor("t1", t1);
net.PutUniTensor("t2", t2);
net.PutUniTensor("t3", t3);

cout << net;
