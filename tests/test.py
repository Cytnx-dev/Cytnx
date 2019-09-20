import cytnx
from cytnx import linalg




#=====================
# Storage 
#=====================
# testing infrastructure--------
a = cytnx.Storage(10,cytnx.Type.Double);
bbb = a.to(cytnx.Device.cpu)
#ccc = a.to(cytnx.Device.cuda+0)

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
print(Ta)
Ta[1,:,0] = ele
print(Ta)
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


##===========================
## Generator
##===========================

Tn = cytnx.zeros(10)
Tn2 = cytnx.zeros(10,dtype=cytnx.Type.Float)
Tn3 = cytnx.zeros(10,device=cytnx.Device.cpu)
print(Tn)
print(Tn2)
print(Tn3)

Tna = cytnx.zeros((2,3))
Tn2a = cytnx.zeros((2,3),dtype=cytnx.Type.Float)
Tn3a = cytnx.zeros((2,3),device=cytnx.Device.cpu)
print(Tna)
print(Tn2a)
print(Tn3a)

##=============================
## Bond
##=============================
bd_in = cytnx.Bond(10,cytnx.bondType.BD_BRA);
print(bd_in)

bd_sym = cytnx.Bond(3,cytnx.bondType.BD_KET,\
                        [[0,2],[1,2],[1,3]],\
                        [cytnx.Symmetry.Zn(2),\
                         cytnx.Symmetry.U1()])
print(bd_sym)

print(bd_sym == bd_sym)
bd_1 = cytnx.Bond(3)
bd_2 = cytnx.Bond(2)
bd_3 = cytnx.Bond(4)

U = cytnx.UniTensor([bd_1,bd_2,bd_3],Rowrank=2,dtype=cytnx.Type.Double)
U.print_diagram()
U.permute_(0,2,1,Rowrank=1)
U.print_diagram()
print(U)
X = U[0,:,:]
X.print_diagram()

U.reshape_(6,-1)
U.print_diagram()




