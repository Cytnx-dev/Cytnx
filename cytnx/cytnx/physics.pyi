"""
physics related.
"""
from __future__ import annotations
import cytnx.cytnx
import typing
__all__: list[str] = ['pauli', 'spin']
def pauli(Comp: str, device: typing.SupportsInt | typing.SupportsIndex = -1) -> cytnx.cytnx.Tensor:
    ...
def spin(S: typing.SupportsFloat | typing.SupportsIndex, Comp: str, device: typing.SupportsInt | typing.SupportsIndex = -1) -> cytnx.cytnx.Tensor:
    ...
