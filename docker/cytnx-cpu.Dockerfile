### Generator Stage
FROM python:3.12-slim AS generator
WORKDIR /build

# Use conda cytnx package or build from source
ARG cytnx_conda="OFF"

# Conda packages version config
ARG cytnx_ver="1"
ARG py_ver="3.12"
ARG compilers_ver=""
ARG make_ver=""
ARG cmake_ver=""

# Pass to environment
ENV cytnx_conda=${cytnx_conda}
ENV cy_ver=${cy_ver}
ENV py_ver=${py_ver}
ENV compilers_ver=${compilers_ver}
ENV make_ver=${make_ver}
ENV cmake_ver=${cmake_ver}

# Set Cytnx compilation build config for build script
COPY docker/build_config.env .
RUN export $(cat build_config.env | xargs)

COPY docker/generate_conda_deps.py docker/generate_build_cytnx.py .
RUN python3 generate_conda_deps.py
RUN python3 generate_build_cytnx.py


### Build stage
FROM continuumio/miniconda3:v25.11.1 AS builder
WORKDIR /opt

# Run conda dependency install script
COPY --from=generator /build/install_conda_deps.sh /build/build_cytnx.sh .
RUN bash install_conda_deps.sh && conda install -y -c conda-forge conda-pack

# Copy cytnx source code
# Dockerfile should be built from the root of the repo!!
COPY . ./Cytnx

# CMake configure, build and install Cytnx
RUN bash build_cytnx.sh

CMD ["/bin/bash"]

# 2. Pack the environment into a standalone archive
# RUN conda-pack -p /opt/conda -o /tmp/env.tar.gz && \
#     mkdir /runtime && cd /runtime && tar -xzf /tmp/env.tar.gz && \
#     rm /tmp/env.tar.gz


### Runtime Stage
# FROM debian:bookworm-slim AS runtime
# WORKDIR /app
#
# # Install minimal system libraries (OpenMP, etc. required by C++ apps)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*
#
# # Copy the packed environment from the builder
# COPY --from=builder /runtime /opt/venv
#
# # Set environment paths so Python and Cytnx are found
# ENV PATH="/opt/venv/bin:$PATH"
# ENV LD_LIBRARY_PATH="/opt/venv/lib:$LD_LIBRARY_PATH"
#
# # Test if it works
# # RUN python3 -c "import cytnx; print('Cytnx version:', cytnx.__version__)"
#
# CMD [ "/bin/bash" ]
