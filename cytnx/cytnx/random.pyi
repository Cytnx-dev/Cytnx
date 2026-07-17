"""
random related.
"""
from __future__ import annotations
import collections.abc
import cytnx.cytnx
import typing
__all__: list[str] = ['normal', 'normal_', 'uniform', 'uniform_']
@typing.overload
def normal(Nelem: typing.SupportsInt | typing.SupportsIndex, mean: typing.SupportsFloat | typing.SupportsIndex, std: typing.SupportsFloat | typing.SupportsIndex, device: typing.SupportsInt | typing.SupportsIndex = -1, seed: typing.SupportsInt | typing.SupportsIndex = -1, dtype: typing.SupportsInt | typing.SupportsIndex = 3) -> cytnx.cytnx.Tensor:
    ...
@typing.overload
def normal(Nelem: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], mean: typing.SupportsFloat | typing.SupportsIndex, std: typing.SupportsFloat | typing.SupportsIndex, device: typing.SupportsInt | typing.SupportsIndex = -1, seed: typing.SupportsInt | typing.SupportsIndex = -1, dtype: typing.SupportsInt | typing.SupportsIndex = 3) -> cytnx.cytnx.Tensor:
    ...
@typing.overload
def normal_(Tin: cytnx.cytnx.Tensor, mean: typing.SupportsFloat | typing.SupportsIndex, std: typing.SupportsFloat | typing.SupportsIndex, seed: typing.SupportsInt | typing.SupportsIndex = -1) -> None:
    ...
@typing.overload
def normal_(Sin: cytnx.cytnx.Storage, mean: typing.SupportsFloat | typing.SupportsIndex, std: typing.SupportsFloat | typing.SupportsIndex, seed: typing.SupportsInt | typing.SupportsIndex = -1) -> None:
    ...
@typing.overload
def uniform(Nelem: typing.SupportsInt | typing.SupportsIndex, low: typing.SupportsFloat | typing.SupportsIndex, high: typing.SupportsFloat | typing.SupportsIndex, device: typing.SupportsInt | typing.SupportsIndex = -1, seed: typing.SupportsInt | typing.SupportsIndex = -1, dtype: typing.SupportsInt | typing.SupportsIndex = 3) -> cytnx.cytnx.Tensor:
    ...
@typing.overload
def uniform(Nelem: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], low: typing.SupportsFloat | typing.SupportsIndex, high: typing.SupportsFloat | typing.SupportsIndex, device: typing.SupportsInt | typing.SupportsIndex = -1, seed: typing.SupportsInt | typing.SupportsIndex = -1, dtype: typing.SupportsInt | typing.SupportsIndex = 3) -> cytnx.cytnx.Tensor:
    ...
@typing.overload
def uniform_(Tin: cytnx.cytnx.Tensor, low: typing.SupportsFloat | typing.SupportsIndex = 0.0, high: typing.SupportsFloat | typing.SupportsIndex = 1.0, seed: typing.SupportsInt | typing.SupportsIndex = -1) -> None:
    ...
@typing.overload
def uniform_(Sin: cytnx.cytnx.Storage, low: typing.SupportsFloat | typing.SupportsIndex = 0.0, high: typing.SupportsFloat | typing.SupportsIndex = 1.0, seed: typing.SupportsInt | typing.SupportsIndex = -1) -> None:
    ...
