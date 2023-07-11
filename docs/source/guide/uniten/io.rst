Save/Load
-----------------
Cyntx provides a way to save/read UniTensors to/from a file.

Save a UniTensor
*****************
To save a Tensor to a file, simply call **UniTensor.Save(filepath)**. 

* In Python:

.. code-block:: python 
    :linenos:
    
    # Create an untagged unitensor and save
    T1 = cytnx.UniTensor(cytnx.zeros([4,4]), rowrank=1)
    T1.set_labels(["a","b"])
    T1.set_name("Untagged_Unitensor")
    T1.Save("Untagged_ut")

    # Create an unitensor with symmetry and save
    bd = cytnx.Bond(cytnx.BD_IN,[[1],[0],[-1]],[1,2,1])
    T2 = cytnx.UniTensor([bd, bd.redirect()], rowrank=1)
    T2.put_block(cytnx.ones([2,2]),1)
    T2.set_labels(["a","b"])
    T2.set_name("symmetric_Unitensor")
    T2.Save("sym_ut")


This will save UniTensors *T1* and *T2* to the current directory as **Untagged_ut.cytnx** and **sym_ut.cytnx**, with *.cytnx* as file extension.


Load a UniTensor
******************
Now, let's load the UniTensor from the file. 

* In Python:

.. code-block:: python
    :linenos:
    
    T1_ = cytnx.UniTensor.Load("Untagged_ut.cytnx")
    print(T1_.labels())
    print(T1_)

    T2_ = cytnx.UniTensor.Load("sym_ut.cytnx")
    print(T2_.labels())
    print(T2_)


Output>>

.. code-block:: text

    ['a', 'b']
    -------- start of print ---------
    Tensor name: Untagged_Unitensorexp(x)!
    is_diag    : False
    contiguous : True

    Total elem: 16
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4,4)
    [[0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 ]
    [0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 ]
    [0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 ]
    [0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 ]]




    ['a', 'b']
    -------- start of print ---------
    Tensor name: symmetric_Unitensor
    braket_form : True
    is_diag    : False
    [OVERALL] contiguous : True
    ========================
    BLOCK [#0]
    |- []   : Qn index 
    |- Sym(): Qnum of correspond symmetry
                  -----------
                  |         |
    [0] U1(1)  -->| 1     1 |-->  [0] U1(1)
                  |         |
                  -----------

    Total elem: 1
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1)
    [[0.00000e+00 ]]

    ========================
    BLOCK [#1]
    |- []   : Qn index 
    |- Sym(): Qnum of correspond symmetry
                  -----------
                  |         |
    [1] U1(0)  -->| 2     2 |-->  [1] U1(0)
                  |         |
                  -----------

    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (2,2)
    [[1.00000e+00 1.00000e+00 ]
    [1.00000e+00 1.00000e+00 ]]

    ========================
    BLOCK [#2]
    |- []   : Qn index 
    |- Sym(): Qnum of correspond symmetry
                   -----------
                   |         |
    [2] U1(-1)  -->| 1     1 |-->  [2] U1(-1)
                   |         |
                   -----------

    Total elem: 1
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1)
    [[0.00000e+00 ]]

In this example we see how the block date and meta information (name, bonds, labels ... ) are kept when we save the UniTensor.

.. toctree::
