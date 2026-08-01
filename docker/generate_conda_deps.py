import os

def generate_conda_deps():
    # Handle Conda Dependencies
    # Pin to latest versions as of 2026-02-05
    conda_deps = {
        "python": os.environ.get("py_ver", "3.12"),
        "compilers": os.environ.get("compilers_ver", ""),
        "make": os.environ.get("make_ver", ""),
        # "cmake": os.environ.get("cmake_ver", "3.26"),
        "cmake": os.environ.get("cmake_ver", ""),
        "numpy": os.environ.get("numpy_ver", ""), # Empty string means latest
        "boost": os.environ.get("boost_ver", ""),
        "libboost": os.environ.get("libboost_ver", ""),
        "pybind11": os.environ.get("pybind11_ver", ""),
        "openblas": os.environ.get("openblas_ver", ""),
        "arpack": os.environ.get("arpack_ver", ""),
    }
    
    # Standard deps without specific version args in your Dockerfile
    extra_deps = ["git", "beartype", "gtest", "benchmark"]
    
    conda_cmd = ["conda", "install", "-y", "-c", "conda-forge"]
    for pkg, ver in conda_deps.items():
        conda_cmd.append(f"{pkg}={ver}" if ver else pkg)
    conda_cmd.extend(extra_deps)

    # Directly install conda cytnx package if cytnx_conda="ON"
    install_cytnx_conda = os.environ.get("cytnx_conda", "OFF")
    if install_cytnx_conda == "ON":
        cytnx_ver = os.environ.get("cytnx_ver", "1.0.0")
        conda_cmd = ["conda", "install", "-y", "-c", "kaihsinwu", f"cytnx={cytnx_ver}"]
    elif install_cytnx_conda != "OFF":
        raise ValueError(f"Invalid value for cytnx_conda: {install_cytnx_conda}")

    with open("install_conda_deps.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(" ".join(conda_cmd) + "\n")

if __name__ == "__main__":
    generate_conda_deps()
