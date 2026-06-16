import numpy as np
from scipy.sparse import csr_matrix


# The same sparse operator as above (A[1,100]=4 and A[100,1]=7 in a 1000x1000
# matrix), but now stored in a real sparse format (scipy CSR) instead of
# pre-storing the elements through the deprecated "mv_elem" path. We construct
# the LinOp with type "mv" and override matvec().
class Oper(cytnx.LinOp):
    def __init__(self):
        cytnx.LinOp.__init__(self, "mv", 1000, cytnx.Type.Double, cytnx.Device.cpu)

        rows = [1, 100]
        cols = [100, 1]
        vals = [4.0, 7.0]
        self.A = csr_matrix((vals, (rows, cols)), shape=(1000, 1000))

    def matvec(self, v):
        # scipy performs the sparse matrix-vector product; we only bridge the
        # cytnx <-> numpy buffers. No dense 1000x1000 matrix is ever formed, and
        # the result has the same shape as the input.
        return cytnx.from_numpy(self.A.dot(v.numpy()))


A = Oper()
x = cytnx.arange(1000)
y = A.matvec(x)

print(x[1].item(), x[100].item())
print(y[1].item(), y[100].item())
