import os,sys

f = open("cuArithmetic_internal.cu",'r')
lines = f.readlines()
f.close()


fo = open("cuArithmetic_internal.cu.tmp",'w')

for line in lines:
    if "type==4" in line:
        fo.write(line)
        var = line.split(" ")[-1]
        var = var.replace("cuCpr","cuMod")
        fo.write("            else            "+var)
    else:
        fo.write(line)


fo.close()
