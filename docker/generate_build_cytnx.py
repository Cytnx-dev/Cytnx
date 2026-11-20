import os

def generate_build_cytnx():
    # Handle CMake Build Flags
    # This list maps the Docker ARG names to the desired Shell ENV names
    flags = [
        "CMAKE_INSTALL_PREFIX", "BUILD_PYTHON", "USE_ICPC", "USE_MKL", 
        "USE_OMP", "USE_CUDA", "USE_HPTT", "USE_CUTENSOR", "USE_CUQUANTUM"
    ]

    with open("build_cytnx.sh", "w") as f:
        f.write("#!/bin/bash\n")
        for flag in flags:
            # Hierarchy: Build Arg (env var) > Default Value ("OFF")
            val = os.environ.get(flag, "")
            if val:
                f.write(f'export {flag}="{val}"\n')

    # If install Cytnx from conda, skip CMake build
    install_cytnx_conda = os.environ.get("cytnx_conda", "OFF")
    if install_cytnx_conda == "ON":
        print("cytnx_conda=ON, skipping CMake build of Cytnx.")
        return

    # CMake configure and build Cytnx
    # Assuming source code mounted at ./Cytnx
    with open("build_cytnx.sh", "a") as f:
        f.write("\n")
        f.write("cd Cytnx\n")
        f.write("mkdir build\n")
        f.write("cd build\n")
        f.write("cmake .. \\\n")
        for flag in flags:
            val = os.environ.get(flag, "")
            if val:
                f.write(f'    -D{flag}="${{{flag}}}" \\\n')
        # f.write("    -DCMAKE_BUILD_TYPE=Release\n")
        # f.write("make -j$(nproc)\n")
        f.write("make\n")
        f.write("make install\n")

    with open("build_cytnx.sh", "r") as f:
        print(f.read())

if __name__ == "__main__":
    generate_build_cytnx()

