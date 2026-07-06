import pytest
import cytnx


def test_cytnx_error_type_exists_and_raises():
    assert issubclass(cytnx.CytnxError, RuntimeError)
    with pytest.raises(cytnx.CytnxError):
        cytnx.zeros([2]).reshape(3, 3)  # shape mismatch -> cytnx_error_msg
    with pytest.raises(RuntimeError):  # subclass relationship preserved
        cytnx.zeros([2]).reshape(3, 3)


def test_message_content():
    with pytest.raises(cytnx.CytnxError, match="reshape"):
        cytnx.zeros([2]).reshape(3, 3)


def test_submodule_error_path_translates_too():
    # linalg.InvM on a non-square Tensor raises via cytnx_error_msg deep inside
    # a submodule binding (linalg_py.cpp) -- prove the module-global translator
    # catches it too, not just top-level Tensor methods.
    with pytest.raises(cytnx.CytnxError, match="InvM"):
        cytnx.linalg.InvM(cytnx.ones([2, 3]))
    with pytest.raises(RuntimeError):
        cytnx.linalg.InvM(cytnx.ones([2, 3]))


def test_non_cytnx_errors_are_not_reclassified():
    # pybind11 overload-resolution failure -> plain TypeError, untouched by the
    # cytnx::error translator.
    with pytest.raises(TypeError) as ei:
        cytnx.ones([2]) + "x"
    assert not isinstance(ei.value, cytnx.CytnxError)
    # pybind11 argument-cast failure -> plain RuntimeError; must NOT be
    # reclassified as CytnxError even though CytnxError subclasses RuntimeError.
    with pytest.raises(RuntimeError) as ei2:
        cytnx.zeros([2]).reshape("x")
    assert not isinstance(ei2.value, cytnx.CytnxError)
