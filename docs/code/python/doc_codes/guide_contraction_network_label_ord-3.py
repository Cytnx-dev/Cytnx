N.PutUniTensor("A1",A1,["v1","phy","v2"])
N.PutUniTensor("A2",A2,["v1","phy","v2"])
Tout=N.Launch()
Tout.print_diagram()
