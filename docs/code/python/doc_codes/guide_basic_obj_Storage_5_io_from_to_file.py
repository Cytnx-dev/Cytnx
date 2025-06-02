# read
A = cytnx.Storage(10)
A.fill(10)
print(A)

A.Tofile("S1")

#load
B = cytnx.Storage.Fromfile("S1",cytnx.Type.Double)
print(B)
