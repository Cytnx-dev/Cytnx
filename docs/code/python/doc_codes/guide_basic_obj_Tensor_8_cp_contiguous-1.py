A = cytnx.zeros([2,3,4])
B = A.permute(0,2,1)

print(A.is_contiguous())
print(B.is_contiguous())
