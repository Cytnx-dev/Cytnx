"""Regression test for #673: constructing `cytnx.Storage(size)` could segfault
in C++ before ever touching another `Storage.cpp` symbol.

Root cause: `Storage::Init` (inline, in the header) dispatched dtype
construction through `Storage_init_interface::USIInit[dtype]`, a `static`
function-pointer table populated only as a side effect of the constructor of
a separate global object (`__SII`, defined in `src/backend/Storage.cpp`).
`Storage::Init` only ever touched the `static` table member, never `__SII`
itself, so a translation unit that called `Storage::Init` without also
referencing anything else from `Storage.cpp` (e.g. `operator<<`) had no
guarantee `__SII`'s constructor had already run. When it hadn't,
`USIInit[dtype]` was still zero-initialized and Cytnx called a null function
pointer -- a classic static-initialization-order fiasco.

The fix (`include/backend/Storage.hpp`, `init_storage`) removes the table and
the `__SII` side-effect object entirely, replacing them with a plain function
that folds over `Type_list` to construct the requested `StorageImplementation<T>`
directly -- there is no cross-translation-unit ordering left to depend on.

Caveat: by the time pytest imports `cytnx`, the whole extension module is
already loaded and every static/global object in it has already run its
constructor, so this test cannot reproduce the exact cross-translation-unit
ordering that made the original C++ bug intermittent (that requires a fresh
C++ process, per the issue's own reproduction). What it guards instead is the
*behavior* of the dispatch path the fix introduced: constructing a Storage of
every supported dtype and round-tripping a representative value, so a
regression that reintroduces a missing/broken dispatch entry for some dtype
is still caught here. A C++ test that reproduces the original
translation-unit-ordering scenario is tracked in a follow-up issue, deferred
until #1080 (test-suite reorganization) merges so it doesn't conflict with
that in-flight rewrite.
"""

import pytest

import cytnx
from cytnx import Type


def test_storage_construction_matches_issue_673_repro():
    # The exact reproduction from #673: construct a Storage and never touch
    # any other Storage.cpp symbol (e.g. operator<<) first.
    a = cytnx.Storage(10)
    assert a.size() == 10
    assert a.dtype() == Type.Double


@pytest.mark.parametrize(
    "dtype,value",
    [
        (Type.ComplexDouble, complex(1.5, -2.5)),
        (Type.ComplexFloat, complex(1.5, -2.5)),
        (Type.Double, -3.25),
        (Type.Float, -3.25),
        (Type.Int64, -12345),
        (Type.Uint64, 12345),
        (Type.Int32, -123),
        (Type.Uint32, 123),
        (Type.Int16, -12),
        (Type.Uint16, 12),
    ],
)
def test_storage_construction_every_dtype_dispatches_correctly(dtype, value):
    s = cytnx.Storage(3, dtype)
    assert s.size() == 3
    assert s.dtype() == dtype
    for i in range(3):
        s[i] = value
    assert [s[i] for i in range(3)] == [value, value, value]


def test_storage_construction_bool_dtype_round_trips_true_and_false():
    s = cytnx.Storage(2, Type.Bool)
    assert s.size() == 2
    assert s.dtype() == Type.Bool
    s[0] = False
    s[1] = True
    assert s[0] is False
    assert s[1] is True
