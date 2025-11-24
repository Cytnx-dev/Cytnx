# Create a rank-2 UniTensor which represents a square matrix
uT = cytnx.UniTensor.arange(4*4).reshape(4,4) \
                    .set_rowrank(1) \
                    .relabel(["in","out"]) \
                    .set_name("initial tensor")
# Eigenvalue decomposition
uD, uV = cytnx.linalg.Eig(uT)
# Create uV, uD, uV_inv = Inv(V) with names and labels
uV.relabel_(["in","a"]).set_name("eigenvectors")
uD.relabel_(["a","b"]).set_name("eigenvalues")
uV_inv = cytnx.linalg.InvM(uV).relabel_(["b","out"]) \
                     .set_name("inverted eigenvectors")
# Compare uT with Uv * uD * uV_inv
uT_new = cytnx.Contracts([uV,uD,uV_inv]) \
              .permute(uT.labels()) \
              .set_name("reconstruction from eigenvalues")
diff = uT_new - uT
print(diff.Norm()) # 1.53421e-14
