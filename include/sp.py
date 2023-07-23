import numpy as np
import os,sys

f = open("UniTensor.hpp","r")

for line in f.readlines():
    if "virtual" in line and not "//" in line:
        tmp = line.strip().split("(")[0]

        print(tmp)

f.close()
