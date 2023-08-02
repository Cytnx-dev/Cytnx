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

@add_method(Storage)
def pylist(self):
    if self.dtype() == Type.Double:
        return self.c_pylist_double();
    elif self.dtype() == Type.ComplexDouble:
        return self.c_pylist_complex128();
    elif self.dtype() == Type.Float:
        return self.c_pylist_float();
    elif self.dtype() == Type.ComplexFloat:
        return self.c_pylist_complex64();
    elif self.dtype() == Type.Uint64:
        return self.c_pylist_uint64();
    elif self.dtype() == Type.Int64:
        return self.c_pylist_int64();
    elif self.dtype() == Type.Uint32:
        return self.c_pylist_uint32();
    elif self.dtype() == Type.Int32:
        return self.c_pylist_int32();
    elif self.dtype() == Type.Uint16:
        return self.c_pylist_uint16();
    elif self.dtype() == Type.Int16:
        return self.c_pylist_int16();
    elif self.dtype() == Type.Bool:
        return self.c_pylist_bool();
    else:
        raise ValueError("[ERROR] Storage.pylist: invalid Storage dtype!");
