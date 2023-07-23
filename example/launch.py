import os,sys

exes = [x for x in os.listdir("./") if ".e" in x]

## seperate objects:
Tensor_exes = [ x for x in exes if "Tensor_" in x]
Storage_exes = [ x for x in exes if "Storage_" in x]
Bond_exes = [ x for x in exes if "Bond_" in x]
Accessor_exes = [ x for x in exes if "Accessor_" in x]
Symmetry_exes = [x for x in exes if "Symmetry_" in x]
Network_exes = [x for x in exes if "Network_" in x]
UniTensor_exes = [x for x in exes if "UniTensor_" in x]

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

for texe in Bond_exes:
    output_name = (texe.split(".e")[0]).split("Bond_")[-1] + ".cpp.out"
    os.system("./%s > Bond/%s"%(texe,output_name))
    print(texe)
    print("================")
    os.system("cat Bond/%s"%(output_name))
    print("================")

for texe in Accessor_exes:
    output_name = (texe.split(".e")[0]).split("Accessor_")[-1] + ".cpp.out"
    os.system("./%s > Accessor/%s"%(texe,output_name))
    print(texe)
    print("================")
    os.system("cat Accessor/%s"%(output_name))
    print("================")

for texe in Symmetry_exes:
    output_name = (texe.split(".e")[0]).split("Symmetry_")[-1] + ".cpp.out"
    os.system("./%s > Symmetry/%s"%(texe,output_name))
    print(texe)
    print("================")
    os.system("cat Symmetry/%s"%(output_name))
    print("================")

for texe in Network_exes:
    output_name = (texe.split(".e")[0]).split("Network_")[-1] + ".cpp.out"
    os.system("./%s > Network/%s"%(texe,output_name))
    print(texe)
    print("================")
    os.system("cat Network/%s"%(output_name))
    print("================")

for texe in UniTensor_exes:
    output_name = (texe.split(".e")[0]).split("UniTensor_")[-1] + ".cpp.out"
    os.system("./%s > UniTensor/%s"%(texe,output_name))
    print(texe)
    print("================")
    os.system("cat UniTensor/%s"%(output_name))
    print("================")
