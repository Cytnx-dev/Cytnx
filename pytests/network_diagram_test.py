import os
import shutil

import pytest

import cytnx
import cytnx.Network_conti as netconti


def _loaded_network():
    """A small loaded Network whose labels exercise Diagram's edge drawing."""
    net = cytnx.Network()
    net.FromString(
        [
            "A: 0,1,2",
            "B: 0,3,4",
            "C: 5,1,6",
            "D: 5,7,8",
            "TOUT: 2,3,4;6,7,8",
            "ORDER: (A,B),(C,D)",
        ]
    )
    return net


def test_diagram_raises_when_graphviz_missing(monkeypatch, tmp_path):
    """When graphviz cannot be imported, Network_conti sets ``Graph`` to None and
    Diagram must fail at call time with a clear ModuleNotFoundError rather than at
    import time."""
    monkeypatch.setattr(netconti, "Graph", None)

    net = _loaded_network()
    with pytest.raises(ModuleNotFoundError, match="graphviz is not installed"):
        net.Diagram(outname=str(tmp_path / "net"))


def test_diagram_renders_when_graphviz_available(monkeypatch, tmp_path):
    """When graphviz (the Python package and the system binaries) is available,
    Diagram renders the network to a file."""
    graphviz = pytest.importorskip("graphviz")
    if shutil.which("circo") is None:
        pytest.skip("graphviz system binaries (circo) are not installed")

    # Render to a file instead of launching a viewer, which would hang or fail in
    # a headless CI environment.
    monkeypatch.setattr(graphviz.Graph, "view", graphviz.Graph.render)

    net = _loaded_network()
    outname = str(tmp_path / "net")
    net.Diagram(outname=outname)

    assert os.path.exists(outname + ".gv")  # graphviz source
    assert os.path.exists(outname + ".gv.pdf")  # rendered diagram
