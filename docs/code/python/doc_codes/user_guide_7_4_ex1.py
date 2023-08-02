from cytnx import Bond
# This creates an in-going Bond with dimension 10.
bond_1 = Bond(10, cytnx.BD_IN)
print(bond_1)
# If one doesn't specify the Bond type, the default bond type will be
# regular or undirectional.
bond_2 = Bond(10)
print(bond_2)
