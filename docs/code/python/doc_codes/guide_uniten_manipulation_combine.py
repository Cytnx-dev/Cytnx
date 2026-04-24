bd1 = cytnx.Bond(cytnx.BD_IN,[[1],[-1]],[1,1])
bd2 = cytnx.Bond(cytnx.BD_IN,[[1],[-1]],[1,1])
bd3 = cytnx.Bond(cytnx.BD_OUT,[[2],[0],[0],[-2]],[1,1,1,1])

ut = cytnx.UniTensor([bd1,bd2,bd3],rowrank=2) \
          .relabel(["a", "b", "c"]).set_name("uT")

print(ut)
ut.combineBonds([0,1], True)
print(ut)
