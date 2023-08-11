# Create an untagged unitensor and save
T1 = cytnx.UniTensor(cytnx.zeros([4,4]), 
                     rowrank=1, 
                     labels=["a","b"], 
                     name="Untagged_Unitensor")
T1.Save("Untagged_ut")

# Create an unitensor with symmetry and save
bd = cytnx.Bond(cytnx.BD_IN,[[1],[0],[-1]],[1,2,1])
T2 = cytnx.UniTensor([bd, bd.redirect()], 
                     rowrank=1, 
                     labels=["a","b"], 
                     name="symmetric_Unitensor")
T2.put_block(cytnx.ones([2,2]),1)
T2.Save("sym_ut")
