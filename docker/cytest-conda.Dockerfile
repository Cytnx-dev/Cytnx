FROM ubuntu:latest
WORKDIR /opt

ARG PY_VER="3.9"

# --- 1. System Dependencies & Download (Separate RUNs for caching)
RUN apt-get update && apt-get install -y curl wget vim bash
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

# --- 2. Conda Installation, Environment Setup, and Package Install (Crucially, one combined RUN)
RUN bash Miniforge3-$(uname)-$(uname -m).sh -p /opt/conda -b && \
    rm Miniforge3-$(uname)-$(uname -m).sh && \
    \
    # Initialize the shell for Conda (using the profile script)
    . /opt/conda/etc/profile.d/conda.sh && \
    \
    # Create the environment
    conda create -n cytnx python=${PY_VER} _openmp_mutex=*=*_llvm -y && \
    \
    # Activate and install (must be in the same RUN command)
    conda activate cytnx && \
    conda install -c kaihsinwu cytnx=1 -y && \
    # Clean up Conda files to keep the image small
    conda clean --all -f -y && \
    conda install -c conda-forge gxx=14 make cmake -y

# --- 3. Final Environment Configuration
# Set the PATH to include the Conda environment binaries
ENV PATH="/opt/conda/envs/cytnx/bin:/opt/conda/bin:$PATH"

WORKDIR /work

CMD ["/bin/bash"]
