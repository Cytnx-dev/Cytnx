# Create a rank-2 Tensor which represents a square matrix
T = cytnx.arange(4*4).reshape(4,4)
# Eigenvalue decomposition
eigvals, V = cytnx.linalg.Eig(T)
# Create UniTensors corresponding to V, D, Inv(V), T
uV=cytnx.UniTensor(V, labels=["a","b"], name="uV")
uD=cytnx.UniTensor(eigvals, is_diag=True, labels=["b","c"], name="uD")
uV_inv=cytnx.UniTensor(cytnx.linalg.InvM(V), labels=["c","d"], name="Inv(uV)")
uT=cytnx.UniTensor(T, labels=["a","d"], name="uT")
# Compare uT with Uv * uD * uV_inv
diff = cytnx.Contracts([uV,uD,uV_inv]) - uT
print(diff.Norm()) # 1.71516e-14