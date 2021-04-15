Create UniTensor
-----------------
As mentioned in the intro, a UniTensor = Block(s) + Bond(s) + Label(s). For which Block(s) are the place holder for data, while Bond(s) and Label(s) are the meta data that describe the properties of the UniTensor. 


Convert from Tensor 
*********************

Before going into more complicated UniTensor structure, first of all let's start with the most simple example, for which we convert a Tensor into a UniTensor. In the following, let's use a simple rank-3 tensor as example. The tensor notation (diagram) looks like:

.. image:: image/untag.png
    :width: 200
    :align: center


.. code-block:: python
    :linenos:

    import cytnx as cy

    # create a rank-3 tensor with shape [2,3,4]
    T = cy.arange(2*3*4).reshape(2,3,4) 

    # convert to UniTensor:
    uT = cy.UniTensor(T,rowrank=1)

    
Here, we simply convert a Tensor **T** into a UniTensor **uT** simply by wrapping it with constructor *cy.UniTensor()*. An additional argument *rowrank* is also provided, which we will get back to it in a moment but now, let's use **print_diagram()** to visualize the UniTensor in a more straightforward way as a diagram: 


.. code-block:: python 
    :linenos:
        
    uT.print_diagram()

Output >> 

.. code-block:: text
    
    tensor Name : 
    tensor Rank : 3
    block_form  : false
    is_diag     : False
    on device   : cytnx device: CPU
                -------------      
               /             \     
         0 ____| 2         3 |____ 1  
               |             |     
               |           4 |____ 2  
               \             /     
                -------------      




There are a lot of information provided in this output which we will explain in details:

1. **Bonds:** They are attach to the left side and/or right side of the center square (which the left/right side is associate to the *rowrank* property, we will get back to this later). The number of bonds indicates the rank of the UniTensor, which also indicates in the second line *tensor Rank*. 

    **Ex:** 
        Here, we have three bonds, indicates it's a rank-3 UniTensor.

2. **Labels&dimensions:** The number on the outside of each bond represent the *label* of that bond, and the numbers indicate the *dimension* (number of elements) of each bond. 

    **Ex:**
        * The bond on the left side   has dimension=2 and label=0.
        * The bond on the upper-right has dimension=3 and label=1.
        * The bond on the lower-right has dimension=4 and label=2. 

3. **tensor name:** The name (alias) of the UniTensor. User can give UniTensor a name using **UniTensor.set_name()** 

.. code-block:: python 
    :linenos:

    uT.set_name("tensor uT")
    print(uT.name())


Output >>
 
.. code-block:: text
    :emphasize-lines: 1

    Tensor name: tensor uT
    braket_form : False
    is_diag    : False

    Total elem: 24
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (2,3,4)
    [[[0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 ]
      [0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 ]
      [0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 ]]
     [[0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 ]
      [0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 ]
      [0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 ]]]


.. tip::

    You can use **UniTensor.name()** to get the name property of the UniTensor.  


4. **on device:** This indicates the data of current UniTensor is on which device (cpu or gpu) 


.. tip::

    Similar to **cytnx.Tensor**, one can use **.to()** to move a UniTensor between devices! 


Construct from Scratch
*************************    














Change Labels
----------------
To change the labels associate to bond(s), we can use **UniTensor.set_label(index, new_label)** or **UniTensor.set_labels(new_labels)**. Note that the label should be integer, and cannot have duplicate labels *within* a same UniTensor:

.. code-block:: python 
    :linenos:

    uT.set_label(1,-9)
    uT.print_diagram()


    uT.set_labels([-8,-10,-999])
    uT.print_diagram()

Output >>

.. code-block:: text

    tensor Name : tensor uT
    tensor Rank : 3
    block_form  : false
    is_diag     : False
    on device   : cytnx device: CPU
                -------------      
               /             \     
         0 ____| 2         3 |____ -9 
               |             |     
               |           4 |____ 2  
               \             /     
                -------------   


    tensor Name : tensor uT
    tensor Rank : 3
    block_form  : false
    is_diag     : False
    on device   : cytnx device: CPU
                -------------      
               /             \     
        -8 ____| 2         3 |____ -10
               |             |     
               |           4 |____ -999
               \             /     
                -------------   











.. toctree::
