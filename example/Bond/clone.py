from cytnx import *
from cytnx.cytnx_extension import *

bd_a = Bond(10)
print(bd_a)

bd_b = bd_a;
bd_c = bd_a.clone();

print(bd_b is bd_a) #true
print(bd_c is bd_a) #false
print(bd_b == bd_a) #true
print(bd_c == bd_a) #true
