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
    


"""
for texe in Storage_exes:
    output_name = (texe.split(".e")[0]).split("Storage_")[-1] + ".cpp.out"
    os.system("./%s > Storage/%s"%(texe,output_name))
    print(texe)
    print("================") 
    os.system("cat Storage/%s"%(output_name))
    print("================") 
"""
