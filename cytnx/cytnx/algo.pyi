"""
algorithm related.
"""
from __future__ import annotations
import collections.abc
import cytnx.cytnx
import typing
__all__: list[str] = ['Concatenate', 'Hsplit', 'Hstack', 'Sort', 'Vsplit', 'Vstack']
def Concatenate(T1: cytnx.cytnx.Tensor, T2: cytnx.cytnx.Tensor) -> cytnx.cytnx.Tensor:
    ...
def Hsplit(Tin: cytnx.cytnx.Tensor, dims: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> list[cytnx.cytnx.Tensor]:
    ...
def Hstack(Tlist: collections.abc.Sequence[cytnx.cytnx.Tensor]) -> cytnx.cytnx.Tensor:
    ...
def Sort(Tn: cytnx.cytnx.Tensor) -> cytnx.cytnx.Tensor:
    ...
def Vsplit(Tin: cytnx.cytnx.Tensor, dims: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> list[cytnx.cytnx.Tensor]:
    ...
def Vstack(Tlist: collections.abc.Sequence[cytnx.cytnx.Tensor]) -> cytnx.cytnx.Tensor:
    ...
