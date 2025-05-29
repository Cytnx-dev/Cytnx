T1_ = cytnx.UniTensor.Load("Untagged_ut.cytnx")
print(T1_.labels())
print(T1_)

T2_ = cytnx.UniTensor.Load("sym_ut.cytnx")
print(T2_.labels())
print(T2_)
