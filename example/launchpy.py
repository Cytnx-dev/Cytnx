import os,sys
from io import StringIO
import contextlib


@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

exes = [x for x in os.listdir("./") if ".e" in x]

## seperate objects:
Tensor_exes = [ x for x in os.listdir("./Tensor") if ".py" in x and not "out" in x]
Storage_exes = [ x for x in os.listdir("./Storage") if ".py" in x and not "out" in x]
Bond_exes = [ x for x in os.listdir("./Bond") if ".py" in x and not "out" in x]
Accessor_exes = [ x for x in os.listdir("./Accessor") if ".py" in x and not "out" in x]
Symmetry_exes = [ x for x in os.listdir("./Symmetry") if ".py" in x and not "out" in x]
Network_exes = [ x for x in os.listdir("./Network") if ".py" in x and not "out" in x]
UniTensor_exes = [ x for x in os.listdir("./UniTensor") if ".py" in x and not "out" in x]
LinOp_exes = [ x for x in os.listdir("./LinOp") if ".py" in x and not "out" in x]
## generate output
for texe in Tensor_exes:
    print(texe)
    print("================")
    output_name = texe + ".out"
    os.system("python %s > %s"%(os.path.join("./Tensor",texe),os.path.join("./Tensor",output_name)))

## generate output
for texe in Storage_exes:
    print(texe)
    print("================")
    output_name = texe + ".out"
    os.system("python %s > %s"%(os.path.join("./Storage",texe),os.path.join("./Storage",output_name)))

## generate output
for texe in Bond_exes:
    print(texe)
    print("================")
    output_name = texe + ".out"
    os.system("python %s > %s"%(os.path.join("./Bond",texe),os.path.join("./Bond",output_name)))

## generate output
for texe in Accessor_exes:
    print(texe)
    print("================")
    output_name = texe + ".out"
    os.system("python %s > %s"%(os.path.join("./Accessor",texe),os.path.join("./Accessor",output_name)))

## generate output
for texe in Symmetry_exes:
    print(texe)
    print("================")
    output_name = texe + ".out"
    os.system("python %s > %s"%(os.path.join("./Symmetry",texe),os.path.join("./Symmetry",output_name)))

## generate output
for texe in Network_exes:
    print(texe)
    print("================")
    output_name = texe + ".out"
    os.system("python %s > %s"%(os.path.join("./Network",texe),os.path.join("./Network",output_name)))

## generate output
for texe in UniTensor_exes:
    print(texe)
    print("================")
    output_name = texe + ".out"
    os.system("python %s > %s"%(os.path.join("./UniTensor",texe),os.path.join("./UniTensor",output_name)))

## generate output
for texe in LinOp_exes:
    print(texe)
    print("================")
    output_name = texe + ".out"
    os.system("python %s > %s"%(os.path.join("./LinOp",texe),os.path.join("./LinOp",output_name)))


"""
for texe in Storage_exes:
    output_name = (texe.split(".e")[0]).split("Storage_")[-1] + ".cpp.out"
    os.system("./%s > Storage/%s"%(texe,output_name))
    print(texe)
    print("================")
    os.system("cat Storage/%s"%(output_name))
    print("================")
"""
