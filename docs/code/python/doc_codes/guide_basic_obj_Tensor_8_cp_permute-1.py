A = cytnx.zeros([2,3,4])
B = A.permute(0,2,1)

print(A)
print(B)

print(B is A)
