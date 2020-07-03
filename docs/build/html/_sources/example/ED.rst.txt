Exact Diagonalization
----------------------
Here, let's consider the example of a 1D transverse field Ising model with Hamiltonain defines as 

.. math::

    H = -\sum_{\left<ij\right>}\sigma^{z}_i\sigma^{z}_j - h\sum_i \sigma^{x}_i

where :math:`\left<ij\right>` denotes nearest neighbor interaction, and :math:`\sigma^{x,z}` is the pauli-matrices. 

The model undergoes a phase transition  at :math:`h_c = 1` with avoid level crossing. For further information, see *insert url!*

Generally, the native way to calculate the energy specturm of this Hamiltonian is through the product of local pauli-matrices. Hoever, the size of this many-body Hamiltonian growth exponentially with size of chain :math:`L` as :math:`2^{L}`. It is not pratical to store this large Hamiltonain. 

Notice that this many-body Hamiltonain is very sparse, with lots of elements are zero. We can use LinOp to represent this Hamiltonain, and call **Lanczos_ER** to get the low-level energies. 







