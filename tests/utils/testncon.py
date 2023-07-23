import numpy as np
from ncon import ncon

# This program need output.txt input as stdin
# Please redirect the stdout to answer.txt
t = None
n = None
m = None
b = None
bonds = []
links = []
uTs = []
n = int(input())

for i in range(n):
    b = int(input())
    bond = list(map(int,input().strip().split(" ")))
    # for j in range(b):
    #     t = int(input())
    #     bond.append(t)
    bonds.append(bond)
    T = np.zeros(bond)
    # print(T)
    tot_dim = 1
    for j in range(len(bond)):
        tot_dim *= bond[j]
    T = T.reshape(tot_dim)
    T = np.array(list(map(int,input().strip().split(" "))), dtype=np.float64)
    # for j in range(tot_dim):
    #     t = int(input())
    #     T[j] = t
    T = T.reshape(bond)
    uTs.append(T)

for i in range(n):
    link = np.array(list(map(int,input().strip().split(" "))))
    # for j in range(len(bonds[i])):
    #     t = int(input())
    #     link.append(t)
    links.append(link)

#print(uTs)
#print(links)
#uTs = np.array(uTs,dtype=np.float64)
#links = np.array(links,dtype=np.float64)
res = ncon(uTs, links)

#np.transpose(res, [-1,-2,-3,-4])

res = list(np.array(res).reshape(-1))
for i in range(len(res)):
    print(str(res[i]),end=" ")
# print(res)
