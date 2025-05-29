uT = cytnx.UniTensor(cytnx.ones([5,5,5,5,5]), rowrank = 3, name="uT")
Q, R = cytnx.linalg.Qr(uT)
Q.set_name("Q")
R.set_name("R")

Q.print_diagram()
R.print_diagram()

# Verify the recomposition
print((cytnx.Contract(Q,R)-uT).Norm())