A = cytnx.arange(24).reshape(2,3,4)
B = A[0,0,1]
C = B.item()
print(B)
print(C)
