import os,sys
import cytnx as cy


class Hising(cy.LinOp):

    def __init__(self,L,J,Hx):
        cy.LinOp.__init__(self,"mv",2**L,cy.Type.Double,cy.Device.cpu)
        ## custom members:
        self.J  = J
        self.Hx = Hx
        self.L  = L

    def SzSz(self,i,j,ipt_id):
        return ipt_id,(1. - 2.*(((ipt_id>>i)&0x1)^((ipt_id>>j)&0x1)))

    def Sx(self,i,ipt_id):
        out_id = ipt_id^((0x1)<<i)
        return out_id,1.0

    ## let's overload this with custom operation:
    def matvec(self,v):
        out = cy.zeros(v.shape()[0],v.dtype(),v.device());
        for a in range(v.shape()[0]):
            for i in range(self.L):
                oid,amp = self.SzSz(i,(i+1)%self.L,a)
                out[oid] += amp*self.J*v[a]

                oid,amp = self.Sx(i,a)
                out[oid] += amp*(-self.Hx)*v[a]
        return out

if __name__ == '__main__':
    L = 4
    J = 1
    Hx = 0.3
    H = Hising(L,J,Hx)
    v = cy.ones(16)
    print(cy.linalg.Lanczos(Hop = H, Tin = v, method = "ER", max_krydim = 2))
