Lanczos solver
----------------
In Cytnx, an eigenvalue problem can be solved for a custom linear operator defined using the **LinOp** class by using the Lanczos iterative solver.

For this, you can pass a **LinOp** or any of its child classes to **linalg.Lanczos_ER**:

.. py:function:: Lanczos_ER(Hop, k=1, is_V=true, maxiter=10000,\
                CvgCrit=1.0e-14, is_row=false, Tin=Tensor(), max_krydim=4)
    
    Perform Lanczos for hermitian/symmetric matrices or linear function.
    
    :param LinOp Hop: the Linear Operator defined by LinOp class or its inheritance.
    :param uint64 k: the number of lowest k eigenvalues
    :param bool is_V: if set to true, the eigenvectors will be returned
    :param uint64 maxiter: the maximum number of iteration steps for each k
    :param double CvgCrit: the convergence criterion of the energy
    :param bool is_row: whether the returned eigenvectors should be in row-major form
    :param Tensor Tin: the initial vector, should be a Tensor with rank-1
    :param uint32 max_krydim: the maximum Krylov subspace dimension for each iteration
    :return: [eigvals (Tensor), eigvecs (Tensor)(option)]
    :rtype: vector<Tensor> (C++ API)/list of Tensor(python API) 


For example, we consider a simple example where we wrap a (4x4) matrix inside a custom operator. We can easily generalize the **matvec** to be any custom sparse structure. 


* In Python:

.. code-block:: python
    :linenos:
    
    class MyOp(cytnx.LinOp):
        def __init__(self):
            cytnx.LinOp.__init__(self,"mv",4)

        def matvec(self,v):
            A = cytnx.arange(16).reshape(4,4)
            A += A.permute(1,0)
            return cytnx.linalg.Dot(A,v)


    op = MyOp()

    v0 = cytnx.arange(4) # trial state
    ev = cytnx.linalg.Lanczos_ER(op,k=1,Tin=v0)

    print(ev[0]) #eigenval
    print(ev[1]) #eigenvec


* In C++:

.. code-block:: c++
    :linenos:

    using namespace cytnx;
    class MyOp: public LinOp{

        MyOp(): LinOp("mv",4){}

        Tensor matvec(const Tensor &v) override{
            auto A = arange(16).reshape(4,4);
            A += A.permute(1,0);
            return linalg::Dot(A,v);
        }

    };

    auto op = MyOp();

    auto v0 = arange(4); // trial state
    auto ev = linalg::Lanczos_ER(&op,1, true, 10000,1.0e-14, false,v0);

    cout << ev[0] << endl; //eigenval
    cout << ev[1] << endl; //eigenvec

Output >>

.. code-block:: text


    Total elem: 1
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1)
    [-7.43416e+00 ]


    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4)
    [-7.98784e-01 -3.77788e-01 8.64166e-02 4.64205e-01 ]



.. Note::

    1. The ER stand for explicitly restarted. The Lanczos method used is based on :lanczos-er:`this reference <>` which can reproduce the degenerate correctly. 

    2. The Lanczos solver only works for symmetric/Hermitian operators.

    3. In cases where the operator is small, try to reduce the max_krydim to get a correct convergence.

.. seealso::

    The solver is used in the example :ref:`SectionED` for the one dimensional transverse field Ising model. 

