"""Helper for importing sibling benchmark scripts from the pytest-benchmark
test modules under `cytnx_bench/`, `tenpy_bench/`, and `quimb_bench/`.

Several of those scripts share a basename across the three library
directories (`dmrg_dense.py`, `dmrg_symmetric.py`, `tebd.py`,
`variational_manual_grad.py`), so a plain `from dmrg_dense import run_one`
caches the module under the bare name `dmrg_dense` in `sys.modules` --
whichever test file imports a given basename first "wins", and every other
test file silently reuses that cached module instead of loading its own
sibling script.
"""
import importlib.util
import os


def load_sibling_module(test_file, module_name):
    test_dir = os.path.dirname(os.path.abspath(test_file))
    module_path = os.path.join(test_dir, f"{module_name}.py")
    unique_name = f"{os.path.basename(test_dir)}.{module_name}"
    spec = importlib.util.spec_from_file_location(unique_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
