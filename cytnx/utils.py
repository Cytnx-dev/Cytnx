from functools import wraps
import inspect
from inspect import Signature
from beartype.door import is_bearable
# from beartype.typing import Dict, FrozenSet, List, Set, Tuple, Type


class Overload_methods:
    #name: str = None
    #reged_sigs: dict[Signature] = {}

    def __init__(self):
        self.name = None
        self.reged_sigs = {}


    def __call__(self,*args,**kwargs):
        for sig in self.reged_sigs:
            try:
                arguments = sig.bind(*args, **kwargs).arguments

                if all(
                    param.annotation is param.empty or is_bearable(arguments[name],param.annotation)
                    for name, param in sig.parameters.items()
                    if name in arguments
                ):
                    return self.reged_sigs[sig](*args,**kwargs)
                else:
                    raise TypeError

            except TypeError:
                continue

        err_msg = f"invalid function signature\n"
        err_msg += f"    args:{args} kwargs:{kwargs}\n"
        err_msg += f"Possible signatures: \n"
        for i, sig in enumerate(self.reged_sigs):
            err_msg += f"{i}. " + str(sig) + "\n"


        raise ValueError(err_msg)


    def _add_func(self,func):
        ## register a function:
        in_sig = inspect.signature(func)

        ## checking function name
        if self.name is None:
            self.name = func.__name__
        else:
            if self.name != func.__name__:
                raise ValueError(f"function name does not match!!")


        ## check if signature already exist:
        if in_sig in self.reged_sigs:
            raise ValueError(f"funciton signature already exists!")

        self.reged_sigs[in_sig] = func



def add_ovld_method(cls):
    """
        inject overload (multiple dispatch) methods

        example:

        .. highlight:: python
        .. code-block:: python

            class Myclass:
                key:str = ""

                def __init__(self):
                    pass

            @add_ovld_method(Myclass)
            def foo(self,a:str,b:str):
                self.key = a+b

            @add_ovld_method(Myclass)
            def foo(self,a:list[str]):
                self.key = "".join(a)
                return "ok"
    """
    def decorator(func):
        cls_member = dict(inspect.getmembers(cls))

        if func.__name__ in cls_member:
            # exists:
            obj = cls_member.get("_" + func.__name__)
            if not isinstance(obj,Overload_methods):
                raise TypeError(f"method <{func.__name__}> already exists\n"+
                                f"(either defined before or injected with add_method).\n"
                                f"  Use @add_ovld_method instead for wrapping overload functions\n"
                               )
            obj._add_func(func)
        else:
            ## if not
            obj = Overload_methods()
            obj._add_func(func)
            setattr(cls,"_" + func.__name__,obj)
            def wrapper(self, *args, **kwargs):
                # get name:
                mem = dict(inspect.getmembers(self))
                obj = mem["_" + func.__name__]

                #passing class instance:
                return obj(self,*args,**kwargs)

            setattr(cls,func.__name__,wrapper)

        return func
    return decorator


def add_method(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self,*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator
