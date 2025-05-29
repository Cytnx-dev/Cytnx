A = cytnx.Storage(10)
A.set_zeros()

B = A.astype(cytnx.Type.ComplexDouble)

print(A)
print(B)
