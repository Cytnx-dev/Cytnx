import os,sys


def resolve_yn(strs):
    strs = strs.strip()
    if len(strs)!=1:
        raise ValueError("invalid input. should be Y/N or y/n")
    strs = strs.upper()
    if strs == "Y":
        return True
    elif strs == "N":
        return False
    else:
        raise ValueError("invalid input. should be Y/N or y/n")

def resolve_num(strs,option):
    strs = strs.strip()
    if len(strs)!=1:
        raise ValueError("invalid input. should be integer")
    strs = int(strs)
    if strs in option:
        return strs
    else:
        raise ValueError("invalid input. should be either",strs)
    
def bool2str(bl):
    if bl:
        return "ON"
    else:
        return "OFF"







## list all the major options:
USE_ICPC=False

USE_MKL=False
USE_OMP=False

USE_CUDA=False
USE_CUTT=False
#CUTT_option_noalign=False
CUTT_option_finetune=False


USE_HPTT=False
HPTT_option_AVX=False
HPTT_option_IBM=False
HPTT_option_ARM=False
HPTT_option_finetune=False

BUILD_PYTHON=True

COMPILE_COMMANDS=True

RUN_TESTS=False

PREFIX=None

step_idx = 1
## get install dest 
#======================================
tmp = input("[%d] install destination (default /usr/local/cytnx):"%(step_idx))
tmp = tmp.strip()
if len(tmp)!=0:
    PREFIX=os.path.expanduser(tmp)

print("  >>CMAKE_INSTALL_PREFIX=%s"%(PREFIX))
print("--------------")


step_idx = 2
## use icpc ? 
#======================================
tmp = input("[%d] build use intel icpc compiler (default OFF)? (Y/N)"%(step_idx))
tmp = tmp.strip()
if(len(tmp.strip())!=0):
    USE_ICPC=resolve_yn(tmp)

print("  >>USE_ICPC:",USE_ICPC)
print("--------------")


step_idx = 3
## checking linalg, and openmp.
#======================================
tmp = input("[%d] use mkl as linalg library (default OFF)? (Y/N):"%(step_idx))
if(len(tmp.strip())!=0):
    USE_MKL=resolve_yn(tmp)

print("  >>USE_MKL: ",USE_MKL)
print("--------------")
if(USE_MKL):
    print("    -->[%da] force USE_OMP=True"%(step_idx))
    print("--------------")
else:
    tmp = input("[%da] use openmp accelerate (default OFF)? (Y/N):"%(step_idx))
    if(len(tmp.strip())!=0):
        USE_OMP=resolve_yn(tmp)
    print("  >>USE_OMP:",USE_OMP)
    print("--------------")

step_idx = 4
## checking HPTT:
#======================================
tmp = input("[%d] use hptt library to accelrate tensor transpose (default OFF)? (Y/N):"%(step_idx))
if(len(tmp.strip())!=0):
    USE_HPTT=resolve_yn(tmp)

print("  >>USE_HPTT: ",USE_HPTT)
print("--------------")
if USE_HPTT:
    ## additional options:
    tmp = input("[%da] hptt option(1): fine tune for the native hardware (default OFF)? (Y/N):"%(step_idx))
    if(len(tmp.strip())!=0):
        HPTT_option_finetune=resolve_yn(tmp)
    print("  >>HPTT_ENABLE_FINE_TUNE:",HPTT_option_finetune)
    print("--------------")
    
    tmp = input("[%db] hptt option(2): variant options (1: AVX 2: IBM 3: ARM, default OFF)? (1,2,3 or enter for default):"%(step_idx))
    if(len(tmp.strip())!=0):
        hptttype=resolve_num(tmp,{1,2,3})
        if(hptttype==1):
            HPTT_option_AVX=True
            print("  >>HPTT_ENABLE_ABX:",HPTT_option_AVX)
        elif(hptttype==2):
            HPTT_option_IBM=True
            print("  >>HPTT_ENABLE_IBM:",HPTT_option_IBM)
        elif(hptttype==3):
            HPTT_option_ARM=True
            print("  >>HPTT_ENABLE_ARM:",HPTT_option_ARM)
        else:
            print("  *No additional options for hptt*")
        print("--------------")


step_idx = 5
## checking CUDA:
#======================================
tmp = input("[%d] with GPU (CUDA) support (default OFF)? (Y/N):"%(step_idx))
if(len(tmp.strip())!=0):
    USE_CUDA=resolve_yn(tmp)

print("  >>USE_CUDA: ",USE_CUDA)
print("--------------")
if USE_CUDA:
    ## additional options:
    tmp = input("[%da] cuda option(1): use cutt library to accelerate tensor transpose (default OFF)? (Y/N):"%(step_idx))
    if(len(tmp.strip())!=0):
        USE_CUTT=resolve_yn(tmp)
    print("  >>USE_CUTT:",USE_CUTT)
    print("--------------")
    
    if USE_CUTT:
        ## add-additional options:
        tmp = input("[%da-1] cutt option(1): fine tune for the native hardware (default OFF)? (Y/N):"%(step_idx))
        if(len(tmp.strip())!=0):
            CUTT_option_finetune=resolve_yn(tmp)
        print("  >>CUTT_ENABLE_FINE_TUNE:",CUTT_option_finetune)
        print("--------------")
        
step_idx = 6
## checking PYTHON:
#======================================
tmp = input("[%d] Build python API (default ON)? (Y/N):"%(step_idx))
if(len(tmp.strip())!=0):
    BUILD_PYTHON=resolve_yn(tmp)

print("  >>BUILD_PYTHON: ",BUILD_PYTHON)
print("--------------")

## checking generate compile_commands.json
#======================================
tmp = input("[%d] Generate compile_commands.json for IDE support (default ON)?(Y/N):"%(step_idx))
if(len(tmp.strip())!=0):
    COMPILE_COMMANDS=resolve_yn(tmp)

print("  >>COMPILE_COMMANDS: ",COMPILE_COMMANDS)
print("--------------")

## whether run cytnx tests
#======================================
tmp = input("[%d] Run cytnx tests (default OFF)?(Y/N):"%(step_idx))
if(len(tmp.strip())!=0):
    RUN_TESTS=resolve_yn(tmp)

print("  >>RUN_TESTS: ",RUN_TESTS)
print("--------------")

##=================================================================
print("*************************")
print("  Review install option  ")
print("")

print(" USE_ICPC: ", USE_ICPC)
print(" USE_MKL: ",USE_MKL)
print(" USE_OMP: ",USE_OMP)

print(" USE_HPTT: ",USE_HPTT)
if(USE_HPTT):
    print(" -- HPTT_option: ")
    print("    HPTT_FINE_TUNE: ",HPTT_option_finetune)
    if(HPTT_option_AVX):
        print("    HPTT_ENABLE_ABX:",HPTT_option_AVX)
    if(HPTT_option_IBM):
        print("    HPTT_ENABLE_IBM:",HPTT_option_IBM)
    if(HPTT_option_ARM):
        print("    HPTT_ENABLE_ARM:",HPTT_option_ARM)

print(" USE_CUDA: ",USE_CUDA)
print(" USE_CUTT: ",USE_CUTT)
if(USE_CUTT):
    print(" -- CUTT_option: ")
    print("    CUTT_ENABLE_FINE_TUNE: ",CUTT_option_finetune)

print(" BUILD_PYTHON: ",BUILD_PYTHON)
print(" COMPILE_COMMANDS: ",COMPILE_COMMANDS)
print("*************************")


## generate sh file:
f = open("ainstall.sh",'w')
f.write("rm -rf build\n")
f.write("mkdir build\n")
f.write("cd build\n")
f.write("cmake")
if not PREFIX is None:
    f.write(" -DCMAKE_INSTALL_PREFIX=%s"%(PREFIX))

if(USE_ICPC):
    f.write(" -DUSE_ICPC=on")

if(USE_MKL):
    f.write(" -DUSE_MKL=on")
else:
    if(USE_OMP):
        f.write(" -DUSE_OMP=on")

if(USE_HPTT):
    f.write(" -DUSE_HPTT=on")
    if(HPTT_option_finetune):
        f.write(" -DHPTT_ENABLE_FINE_TUNE=on")        

        
    if(HPTT_option_AVX):
        f.write(" -DHPTT_ENABLE_AVX=on")
    if(HPTT_option_IBM):
        f.write(" -DHPTT_ENABLE_IBM=on")
    if(HPTT_option_ARM):
        f.write(" -DHPTT_ENABLE_ARM=on")

if(USE_CUDA):
    f.write(" -DUSE_CUDA=on")
    if(USE_CUTT):
        f.write(" -DUSE_CUTT=on")
        if(CUTT_option_finetune):
            f.write("-DCUTT_ENABLE_FINE_TUNE=on")
        
if(BUILD_PYTHON):
    f.write(" -DBUILD_PYTHON=on")
else:
    f.write(" -DBUILD_PYTHON=off")

if(COMPILE_COMMANDS):
    f.write(" -DCMAKE_EXPORT_COMPILE_COMMANDS=1")

if(RUN_TESTS):
    f.write(" -DRUN_TESTS=on")

f.write(" ../\n")
f.write("make -j `nproc`\n")
f.write("make install\n")
if(RUN_TESTS):
    f.write("GTEST_COLOR=1 ctest --output-junit junit.xml")

f.close()


