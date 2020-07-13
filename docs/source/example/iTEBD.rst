iTEBD
------------
Time evolution block decimation is one of the most simple and sucessful Tensor network method :cite:`itebd-vidal`. The core concept of this algorithm is to use the imaginary time evolution to find the best variational ansatz, usually in terms of Matrix product state (MPS). 


Here, we use a 1D transverse field Ising model (TFIM) as a simple example to show how to implement iTEBD algorithm in Cytnx and get the infinite system size (variational) ground state. 

Consider the Hamiltonain of TFIM:

.. math::

    H = J\sum_{ij} \sigma^{z}_i\sigma^{z}_j - H_x\sum_i \sigma^{x}_i

where :math:`\sigma^{x,z}` are the pauli matrices. The infinite size ground state can be represent by MPS as variational ansatz, where the *virtual bonds* dimension :math:`\chi` effectively controls the number of variational parameters (shown as orange bonds), and the *physical bonds* dimension :math:`d` is the real physical dimension (shown as the blue bonds. Here, for Ising spins :math:`d=2`). 

Because the system has translational invariant, thus it is legit to choose unit-cell consist with two sites, and the infinite system ground state can be represented with only two sites MPS with local tensors :math:`\Gamma_A` and :math:`\Gamma_B` associate with the schmit basis and :math:`\lambda_A`, :math:`\lambda_B` are the diagonal matrices of Schmidt coefficients as shown in the following:

.. image:: image/itebd_MPS.png
    :width: 600
    :align: center







.. Hint::
    
    The complete example code can be found in Github repo under example/iTEBD folder.


.. bibliography:: ref.itebd.bib
    :cited:
