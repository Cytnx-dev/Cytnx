// A & B share same memory
auto A = cytnx::Storage(10);
auto B = cytnx::Tensor::from_storage(A);

// A & C have different memory
auto C = cytnx::Tensor::from_storage(A.clone());
