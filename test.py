import cytnx

a = cytnx.Storage(10,cytnx.cytnxtype.Double,cytnx.cytnxdevice.cuda);

print(a[3])
print(a.dtype)
print(a.dtype_str)

