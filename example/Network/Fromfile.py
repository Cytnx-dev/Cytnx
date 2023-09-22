import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
import cytnx


N = cytnx.Network("example.net")
print(N)
