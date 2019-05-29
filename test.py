import cytnx



#=====================
# Storage 
#=====================
# testing infrastructure:
a = cytnx.Storage(10,cytnx.Type.Double);

print(a[3])
print(a.dtype)
print(a.dtype_str)
print(a)

a.fill(14)
print(a)
b = a.clone()
print(b is a)
print(b==a)
b[7] = 100
print(b)
c = b.astype(cytnx.Type.Uint64)
print(c)
c.print_info()
