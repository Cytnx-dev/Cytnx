from __future__ import annotations
import typing
__all__: list[str] = ['Ngpus', 'Print_Property', 'cpu', 'cuda', 'getname']
def Print_Property() -> None:
    ...
def getname(arg0: typing.SupportsInt | typing.SupportsIndex) -> str:
    ...
Ngpus: int = 0
cpu: int = -1
cuda: int = 0
