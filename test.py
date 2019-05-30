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

#=====================
# Tensor
#=====================
Ta = cytnx.Tensor((3,4,2),dtype=cytnx.Type.Double)
Ta_shp = Ta.shape
print(Ta_shp)
print(Ta.dtype_str)
print(Ta.device_str)
print(Ta)

Ta.permute_(0,2,1)
Tb = Ta.permute_(1,0,2)
print(Tb is Ta)
Tc = Ta.reshape(12,2)
print(Tc.shape)

