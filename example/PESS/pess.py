import numpy as np 
import cytnx as cy

##
# Author: Kai-Hsin Wu
##

def inv_e(A,clip):
    out = A.clone()
    b = out.get_block_()
    for i in range(b.shape()[0]):
        if(b[i].item()<clip):
            b[i] = 0;
        else:
            b[i] = 1./b[i] 

    return out

## create opeartor:
Sz = cy.physics.spin(0.5,'z')
Sx = cy.physics.spin(0.5,'x')
Sy = cy.physics.spin(0.5,'y')
I = cy.eye(2)

H = cy.linalg.Kron(Sz,Sz) + cy.linalg.Kron(Sx,Sx) + cy.linalg.Kron(Sy,Sy)
H = H.real() # since it's real ,let's truncate imag part.
H = cy.linalg.Kron(H,I)
H.reshape_(2,2,2,2,2,2)
H = H + H.permute(0,2,1,3,5,4) + H.permute(2,0,1,5,3,4)


###################
tau = 1.0
D = 3
maxit = 1000

# Here, we explicitly seperate up/down
# which is the same in this model, but in general they could be different
Hup = H
Hdown = Hup.clone()

## Hamiltonain 
Hu = cy.UniTensor(Hup,rowrank=3)
Hd = cy.UniTensor(Hdown,rowrank=3)

## create 3PESS:
s1 = cy.UniTensor(cy.random.normal([D,D,D],0,0.5),rowrank=0)
s2 = s1.clone()
A = cy.UniTensor(cy.random.normal([D,2,D],0,0.5),rowrank=2)
B = cy.UniTensor(cy.random.normal([D,2,D],0,0.5),rowrank=2)
C = cy.UniTensor(cy.random.normal([D,2,D],0,0.5),rowrank=2)
Ls1 = [cy.UniTensor([cy.Bond(D),cy.Bond(D)],rowrank=1,is_diag=True) for i in range(3)]
for i in Ls1:
    i.put_block(cy.ones(D))
Ls2 = [i.clone() for i in Ls1]


## gates:
eHu = cy.linalg.ExpH(Hu ,-tau)
eHd = cy.linalg.ExpH(Hd,-tau)

## iterator:
Eup_old,Edown_old = 0,0
cov = False
for i in range(maxit):

    ## first do up >>>>>>>>>>>>>>>>>>
    Nup = cy.Network("up.net")
    Nup.PutUniTensors(["A","B","C","L2A","L2B","L2C","eHu","s1"],[A,B,C,Ls2[0],Ls2[1],Ls2[2],eHu,s1]) 
    T = Nup.Launch(True)
    Nrm = cy.Contract(T,T).item();
    T/=np.sqrt(Nrm); #normalize for numerical stability.
    A,B,C,s1,Ls1[0],Ls1[1],Ls1[2] = cy.linalg.Hosvd(T,[2,2,2],True,True,[D,D,D])
    
    ## de-contract Ls'
    Ls2[0].set_labels([-10,A.labels()[0]]); Ls2[0] = 1./Ls2[0];
    Ls2[1].set_labels([-11,B.labels()[0]]); Ls2[1] = 1./Ls2[1];
    Ls2[2].set_labels([-12,C.labels()[0]]); Ls2[2] = 1./Ls2[2];
    A = cy.Contract(Ls2[0],A);
    B = cy.Contract(Ls2[1],B);
    C = cy.Contract(Ls2[2],C);


    T = cy.Contract(cy.Contract(cy.Contract(A,s1),B),C)
    T.set_rowrank(0)
    ## calculate up energy:
    NE = cy.Network("measure.net")
    NE.PutUniTensors(["T","Tt","Op"],[T,T,Hu]);
    Eup = NE.Launch(True).item()/cy.Contract(T,T).item();    

    ## then do down>>>>>>>>>>>>>> 
    Ndown = cy.Network("down.net")
    Ndown.PutUniTensors(["A","B","C","L1A","L1B","L1C","eHd","s2"],[A,B,C,Ls1[0],Ls1[1],Ls1[2],eHd,s2])
    T = Ndown.Launch(True)
    Nrm = cy.Contract(T,T).item();
    T/=np.sqrt(Nrm); #normalize for numerical stability.
    A,B,C,s2,Ls2[0],Ls2[1],Ls2[2] = cy.linalg.Hosvd(T,[2,2,2],True,True,[D,D,D])

    ## de-contract Ls'
    Ls1[0].set_labels([-10,A.labels()[0]]); Ls1[0] = 1./Ls1[0];
    Ls1[1].set_labels([-11,B.labels()[0]]); Ls1[1] = 1./Ls1[1];
    Ls1[2].set_labels([-12,C.labels()[0]]); Ls1[2] = 1./Ls1[2];
    A = cy.Contract(Ls1[0],A);
    B = cy.Contract(Ls1[1],B);
    C = cy.Contract(Ls1[2],C);

    ## calculate down energy:
    T = cy.Contract(cy.Contract(cy.Contract(A,s1),B),C)
    T.set_rowrank(0)
    NE.PutUniTensors(["T","Tt","Op"],[T,T,Hd]);
    Edown = NE.Launch(True).item()/cy.Contract(T,T).item();
   
    ##permute to the current order of bond for next iteration:
    A.permute_([2,1,0]);
    B.permute_([2,1,0]);
    C.permute_([2,1,0]);

    print("[iter%d] Eu: %11.16f Ed: %11.16f"%(i,Eup,Edown))
    if(abs(Eup-Eup_old)<1.0e-14 and abs(Edown-Edown_old)<1.0e-14):
        cov = True;
        break;

    Eup_old = Eup;  
    Edown_old = Edown;


if cov:
    print("[converge]")
else:
    print("[not-converge]")










