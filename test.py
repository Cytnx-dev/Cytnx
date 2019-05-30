import cytnx



#=====================
# Storage 
#=====================
# testing infrastructure--------
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
# infrastructure-------------
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

ele = Ta[0,:,1]
print(ele)
scalar_t = Ta[0,1,1]
print(scalar_t)
val = scalar_t.item()
print(val)

# arithmetic------------
Ta_add = Ta + 3.
Ta_add_2 = 3. + Ta

# now all the arithmetic require contiguous call.
# it will becomes implicity in later version.
Ta.contiguous_() 
                  
Ta_add_3 = Ta + Ta
print(Ta_add)
print(Ta_add_2)
print(Ta_add_3)

Ta_sub = Ta - 3.
Ta_sub_2 = 3. - Ta
Ta_sub_3 = Ta - Ta

Ta_mul = Ta * 3.
Ta_mul_2 = 3. * Ta
Ta_mul_3 = Ta * Ta

Ta_div = Ta/3.
Ta_div_2 = 3./Ta
Ta_div_3 = Ta/Ta

Ta_div_int = Ta/3
Ta_div_int_2 = 3/Ta
Ta_div_int_3 = Ta/Ta
