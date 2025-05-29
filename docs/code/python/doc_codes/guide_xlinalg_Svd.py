T = cytnx.UniTensor(cytnx.ones([5,5,5,5,5]), rowrank = 3)
S, U, Vt = cytnx.linalg.Svd(T)
U.set_name('U')
S.set_name('S')
Vt.set_name('Vt')


T.print_diagram()
S.print_diagram()
U.print_diagram()
Vt.print_diagram()
