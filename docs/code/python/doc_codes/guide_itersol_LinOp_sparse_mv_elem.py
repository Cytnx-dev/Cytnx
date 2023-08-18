class Oper(cytnx.LinOp):

    def __init__(self):
        cytnx.LinOp.__init__(self,"mv_elem",1000)

        self.set_elem(1,100,4.)
        self.set_elem(100,1,7.)

A = Oper();
x = cytnx.arange(1000)
y = A.matvec(x)

print(x[1].item(),x[100].item())
print(y[1].item(),y[100].item())
