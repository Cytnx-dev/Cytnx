FROM ubuntu:18.04
MAINTAINER kaihsinwu


ENV PATH=/miniconda/bin:${PATH}
ENV LD_LIBRARY_PATH=/miniconda/lib:/opt/cytnx:${LD_LIBRARY_PATH}
ENV CPATH=/miniconda/include:/opt/cytnx/include:${CPATH}
ENV PYTHONPATH=/opt/cytnx:${PYTHONPATH}
RUN mkdir /opt/cytnx
COPY cytnx /opt/cytnx/ 

RUN apt-get update\
    && apt-get -y install make curl vim g++-6\
    && curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh\
    && bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b\
    && rm Miniconda3-latest-Linux-x86_64.sh\
    && conda update -y conda\
    && conda install -y numpy\
    && conda install -y -c conda-forge pybind11 openmp\
    && conda install -y -c anaconda mkl-include boost\
    && cd /opt/cytnx\
    && make -Bj4\
    && make pyobj
