from typing import Any
from cytnx import *

def Load(obj: Any, container, name: str, path: str = ""):
    io.c_Load(obj, container, name, path)

# inject into the submodule
obj = io
setattr(obj,"Load",Load)
