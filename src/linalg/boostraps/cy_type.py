DTYPES_SIMPLE = ["cd", "cf", "d", "f", "i64", "u64", "i32", "u32", "i16", "u16", "b"]
DTYPES_FULL = ["cytnx_complex128", "cytnx_complex64", "cytnx_double", "cytnx_float", "cytnx_int64", "cytnx_uint64", "cytnx_int32", "cytnx_uint32", "cytnx_int16", "cytnx_uint16", "bool"]
cuDTYPES_FULL = ["cuDoubleComplex", "cuFloatComplex", "cytnx_double", "cytnx_float", "cytnx_int64", "cytnx_uint64", "cytnx_int32", "cytnx_uint32", "cytnx_int16", "cytnx_uint16", "bool"]
SIGNED = [1,1,1,1,1,0,1,0,1,0,0]

def typeid_promote(typeid1,typeid2):
    if(typeid1 < typeid2):
        if(SIGNED[typeid2] and not SIGNED[typeid1]):
            return typeid1-1; # make it signed
        else:
            return typeid1;
    else:
        if(SIGNED[typeid1] and not SIGNED[typeid2]):
            return typeid2-1;
        else:
            return typeid2


if __name__ == "__main__":

    for t1, t1name in enumerate(DTYPES_FULL):
        for t2, t2name in enumerate(DTYPES_FULL):
            print("%s\t%s\t%s"%(DTYPES_FULL[typeid_promote(t1,t2)],t1name,t2name))
