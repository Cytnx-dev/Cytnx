from __future__ import annotations
import typing
__all__: list[str] = ['Ncpus', 'Ngpus', 'Print_Property', 'cpu', 'cuda', 'getname']
def Print_Property() -> None:
    ...
def getname(arg0: typing.SupportsInt | typing.SupportsIndex) -> str:
    ...
Ncpus: int = 4
Ngpus: int = 0
cpu: int = -1
cuda: int = 0
