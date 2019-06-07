import os,sys

exes = [x for x in os.listdir("./") if ".e" in x]

## seperate objects:
Tensor_exes = [ x for x in exes if "Tensor_" in x]
Storage_exes = [ x for x in exes if "Storage_" in x]



## generate output
for texe in Tensor_exes:
    output_name = (texe.split(".e")[0]).split("Tensor_")[-1] + ".cpp.out"
    os.system("./%s > Tensor/%s"%(texe,output_name))
    print(texe)
    print("================") 
    os.system("cat Tensor/%s"%(output_name))
    print("================") 

for texe in Storage_exes:
    output_name = (texe.split(".e")[0]).split("Storage_")[-1] + ".cpp.out"
    os.system("./%s > Storage/%s"%(texe,output_name))
    print(texe)
    print("================") 
    os.system("cat Storage/%s"%(output_name))
    print("================") 

