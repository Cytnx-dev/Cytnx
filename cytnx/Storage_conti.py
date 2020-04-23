from .utils import *
from cytnx import *

class StorageIterator:
    def __init__(self,sd):
        self._sd = sd;
        self._index = 0;

    def __next__(self):
        if(self._index < len(self._sd)):
            
            result =  self._sd[self._index]
            self._index +=1;
            return result

        else:
            raise StopIteration


@add_method(Storage)
def astype(self, new_type):
    if(self.dtype() == new_type):
        return self

    else:
        return self.astype_different_type(new_type)

@add_method(Storage)
def to(self, device):
    if(self.device() == device):
        return self

    else:
        return self.to_different_device(device)

@add_method(Storage)
def __iter__(self):
    return StorageIterator(self)
