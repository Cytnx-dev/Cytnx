Create UniTensor
------------------
As mentioned in the introduction, a **UniTensor** consists of Block(s), Bond(s) and Label(s). The Block(s) contain the data, while Bond(s) and Label(s) are the meta data that describe the properties of the UniTensor. 

.. image:: image/utcomp.png
    :width: 600
    :align: center




Generally, there are two types of UniTensor types: **un-tagged** and **tagged**, depending on whether the bond has a *direction*. In more advanced applications, a UniTensor may have block diagonal or other more complicated structure when symmetries are involved. In that case, the UniTensor can further be categorized into **non-symmetric** and **symmetric (block form)**:

+-----------+-----------------+-------------------------------+
|           |  non-symmetric  |  symmetric (block-diagonal)   |
+-----------+-----------------+-------------------------------+
| tagged    |     **O**       |            **O**              |
+-----------+-----------------+-------------------------------+
| untagged  |     **O**       |            **X**              |
+-----------+-----------------+-------------------------------+

   
In the following, we will explain how to construct a UniTensor. 


Construct from Tensor 
************************

Before going into more complicated UniTensor structures, let's start with the most simple example. For this, we convert a **cytnx.Tensor** into a UniTensor. This gives us the first type of a UniTensor: an **untagged** UniTensor.  

In the following, we consider a simple rank-3 tensor as an example to give you a glance on some basic properties of UniTensor. The tensor diagram looks like:

.. image:: image/untag.png
    :width: 200
    :align: center

We can convert such a Tensor to a UniTensor:

.. code-block:: python
    :linenos:

    import cytnx as cy

    # create a rank-3 tensor with shape [2,3,4]
    T = cy.arange(2*3*4).reshape(2,3,4) 

    # convert to UniTensor:
    uT = cy.UniTensor(T)

    
Here, the Tensor **T** is converted to a UniTensor **uT** simply by wrapping it with constructor *cy.UniTensor()*. Formally, we can think of this as constructing a UniTensor **uT** with **T** being its *block* (data). 

We can use **print_diagram()** to visualize a UniTensor in a more straightforward way as a diagram: 


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



The information provided by the output is explained in the following:

1. **Bonds:** They are attached to the left side and/or right side of the center square. Now you might wonder why there are bonds going to two sides? In Cytnx, we use a property called **rowrank** which defines this. The first *rowrank* bonds will be considered to direct to the left and the rest will be on the right. We will get back to this property later. For now, let's just assume that rowrank takes an arbitrary integer 0 < rowrank < rank. The number of bonds indicates the rank of the UniTensor, which is also printed in the second line as *tensor Rank*. 

    **Example:** 
        Here, we have three bonds, indicating it is a rank-3 UniTensor.

2. **Labels&dimensions:** The number on the outside of each bond represents the *label* of that bond, and the numbers inside the box indicate the *dimension* (number of elements) of each bond. 

    **Example:**
        * The bond on the left side   has dimension=2 and label="0".
        * The bond on the upper-right has dimension=3 and label="1".
        * The bond on the lower-right has dimension=4 and label="2". 


.. note::

    The order of the bonds are arranged from left to right and up to down. In this example, the bond with label="0" is the first bond (index=0); the bond with label="1" is the second bond (index=1); the bond with label="2" is the 3rd bond (index=2).


.. note::

    The labels of bonds are strings, and therefore text. By default, these labels are set to "0", "1", ... if no label names are defined. These are strings containing the corresponding number. Labels can not be integers. This is because in many APIs a bond can either be addressed by its index number (integer) as for a **cytnx.Tensor** or by its name (string).


3. **tensor name:** The name (alias) of the UniTensor. Users can name a UniTensor with **UniTensor.set_name()**:

.. code-block:: python 
    :linenos:

    uT.set_name("tensor uT")
    print(uT)


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


4. **on device:** This indicates the device where the data of the UniTensor is stored (cpu or gpu). 


.. tip::

    Similar to **cytnx.Tensor**, one can use **.to()** to move a UniTensor between devices! 


.. note::
    
    The dtype and device of a UniTensor depends on the underlying *block* (data) of the UniTensor. 


From scratch
**************  

Next, let's introduce the complete API for constructing a UniTensor from scratch:


.. py:function:: UniTensor(bonds, labels, rowrank, dtype, device, is_diag)
     
    :param List[cytnx.Bond] bonds: list of bonds 
    :param List[string] labels: list of labels associate to each bond 
    :param int rowrank: rowrank used when flattened into a matrix 
    :param cytnx.Type dtype: the dtype of the block(s) 
    :param cytnx.Device device: the device where the block(s) are held 
    :param bool is_diag: whether the UniTensor is diagonal 

The first argument **bonds** is a list of bond objects. These correspond to the *shape* of a **cytnx.Tensor** where the elements in *shape* indicate the dimensions of the bonds. Here, each bond is represent by a **cytnx.Bond** object. In general, **cytnx.Bond** contains three things:

1. The dimension of the bond. 
2. The direction of the bond (it can be BD_REG--undirectional, BD_KET (BD_IN)--inward, BD_BRA (BD_OUT)--outward) 
3. The symmetry and the associate quantum numbers. 

For more details, see the **Bond** section. Here, for simplicity, we will use only the dimension property of a Bond. 

Now let's construct the rank-3 UniTensor with the same shape as in the above example. We assign the three bonds with labels ("a", "b", "c") and also set name to be "uT2 scratch".

.. image:: image/ut2.png
    :width: 300
    :align: center


.. code-block:: python
    :linenos:

    import cytnx as cy
    from cytnx import Bond as bd

    uT2 = cy.UniTensor([bd(2),bd(3),bd(4)],labels=["a","b","c"],rowrank=1).set_name("uT2 scratch")
    uT2.print_diagram()
    print(uT2)

Output >>

.. code-block:: text
    
    -----------------------
    tensor Name : uT2 scratch
    tensor Rank : 3
    block_form  : false
    is_diag     : False
    on device   : cytnx device: CPU
                -------------      
               /             \     
         a ____| 2         3 |____ b
               |             |     
               |           4 |____ c
               \             /     
                -------------  

    Tensor name: uT2 scratch
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


.. note:: 

    The UniTensor will have all the elements in the block initialized with zeros. 