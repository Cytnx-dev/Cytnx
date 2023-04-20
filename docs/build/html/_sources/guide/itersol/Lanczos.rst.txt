Lanczos solver
----------------
Currently (v0.5.5a) Cytnx provide the Lanczos iterative solver that solve the eigen value problem of the custom operator defined using **LinOp** class.

To use, you can pass either the **LinOp** itself or any of it's interitance object to **linalg.Lanczos_ER**

* Lanczos_ER signature:

.. py:function:: Lanczos_ER(Hop, k=1, is_V=true, maxiter=10000,\
                CvgCrit=1.0e-14, is_row=false, Tin=Tensor(), max_krydim=4)
    
    Perform Lanczos for hermitian/symmetric matrices or linear function.
    
    :param LinOp Hop: the Linear Operator defined by LinOp class or it's inheritance.
    :param uint64 k: the number of lowest k eigen values
    :param bool is_V: if set to true, the eigen vectors will be returned
    :param uint64 maxiter: the maximum interation steps for each k
    :param double CvgCrit: the convergence criterion of the energy
    :param bool is_row: whether the return eigen vectors should be in row-major form
    :param Tensor Tin: the initial vector, this should be rank-1
    :param uint32 max_krydim: the maximum krylov subspace dimension for each iteration
    :return: [eigvals (Tensor), eigvecs (Tensor)(option)]
    :rtype: vector<Tensor> (C++ API)/list of Tensor(python API) 


For example, let's consider a simple example of wrapping a (4x4) matrix inside a custom operator, you can easily generalize the **matvec** to be any custom sparse structure. 


* In python:

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

    print(ev[0]) #eigen val
    print(ev[1]) #eigen vec


* In c++:

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

    cout << ev[0] << endl; //eigen val
    cout << ev[1] << endl; //eigen vec

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

    1. The ER stand for explicitly restarted. The Lanczos method is base on :lanczos-er:`This reference <>` which can capture the degenerate correctly. 

    2. Lanczos only work for symmetric/Hermitian operator.

    3. in case where the operator is small, try to reduce the max_krydim to get correct convergence.

.. seealso::

    Examples/Exact diagonalization for example of exact diagonalization calculation in 1D transverse field ising model. 

