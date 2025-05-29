A = cytnx.arange(24).reshape(2,3,4)
print(A.is_contiguous())
print(A)

A.permute_(1,0,2)
print(A.is_contiguous())
print(A)

A.contiguous_()
print(A.is_contiguous())
