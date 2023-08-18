A = cytnx.ones([3,4,5])
B = cytnx.ones([4,5])*2
print(A)
print(B)

A.append(B)
print(A)
