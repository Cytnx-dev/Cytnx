## set compile mode


## compile with CUDA support:
GPU_Enable=1

## compile with Openmp support:
OMP_Enable=1

## compile with MKL library:
MKL_Enable=0

## compile using intel icpc compiler
ICPC_Enable=0

## the cuda path, should provided when GPU_Enable=1
CUDA_PATH=/usr/local/cuda

## compile with debug mode, including more information and check at runtime
DEBUG_Enable=0



## un-commend when distribute by docker  

#DOCKER_MKL=-L/miniconda/lib

