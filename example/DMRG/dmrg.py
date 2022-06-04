import cytnx as cy
from cytnx import tn_algo

def run_DMRG(mpo, mps, Nsweeps, ortho_mps=[], weight=40):
    
    model = tn_algo.DMRG(mpo, mps, ortho_mps, weight);
    model.initialize();
    for xi in range(Nsweeps):
        E = model.sweep();
        print(E);

    return E;
    

Nsites = 10;
chi = 16;
weight = 40;
h = 4;


#construct MPO:
sz = cy.physics.pauli("z").real();
sx = cy.physics.pauli("x").real();
II  = cy.eye(2);

tM = cy.zeros([3,3,2,2]);
tM[0,0] = II;
tM[-1,-1] = II;
tM[0,2] = -h*sx;
tM[0,1] = -sz;
tM[1,2] = sz;
uM = cy.UniTensor(tM);

mpo = cy.tn_algo.MPO();
mpo.assign(Nsites,uM);


#starting DMRG:
mps0 = tn_algo.MPS(Nsites,2,chi);
E0 = run_DMRG(mpo,mps0,Nsweeps=20);

# first excited
mps1 = tn_algo.MPS(Nsites,2,chi);
E1 = run_DMRG(mpo,mps1,Nsweeps=30,ortho_mps=[mps0],weight=60);

# second excited.
mps2 = tn_algo.MPS(Nsites,2,chi);
E2 = run_DMRG(mpo,mps2,Nsweeps=40,ortho_mps=[mps0,mps1],weight=60);

print(E0)
print(E1)
print(E2)
