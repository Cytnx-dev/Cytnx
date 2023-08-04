net.PutUniTensor("T0", A, ["left", "phy", "right"])
net.PutUniTensor("T1", B, ["left", "phy", "right"])
Tout=net.Launch()
Tout.print_diagram()
