class Oper(cytnx.LinOp):
    Loc = []
    Val = []

    def __init__(self):
        cytnx.LinOp.__init__(self,"mv",1000)

        self.Loc.append([1,100])
        self.Val.append(4.)

        self.Loc.append([100,1])
        self.Val.append(7.)

    def matvec(self,v):
        out = cytnx.zeros(v.shape(),v.dtype(),v.device())
        for i in range(len(self.Loc)):
            out[self.Loc[i][0]] += v[self.Loc[i][1]]*self.Val[i]
        return out


A = Oper();
x = cytnx.arange(1000)
y = A.matvec(x)

print(x[1].item(),x[100].item())
print(y[1].item(),y[100].item())
