Bond
=======
A **Bond** is an object that represent the legs of a tensor. It carries informations such as direction, dimension and quantum numbers (if with Symmetry). 

There are in general two types of Bonds: **directional** and **undirectional** depending on whether the bond has direction (pointing inward to or outward from the tensor body) or not. The inward Bond is also defined as **Ket**/**In** type, and the outward Bond is defined as **Bra**/**Out** type, which represent the *Braket* notation in the quantum mechanic: 

.. image:: image/bond.png
    :width: 400
    :align: center

Let's introduce the complete API for constructing a simple Bond (with or without direction)

.. py:function:: Bond(dim, bd_type)
     
    :param int dim: The dimension of the bond.
    :param bondType bd_type: The type (direction) of the bond, can be BD_REG--undirectional, BD_KET--inward (same as BD_IN), BD.BRA--outward (same as BD_OUT)





Symmetry object
**********************

In Cytnx we have the Symmetry as object, it mainly contains the name, type, combine rule and the reverse rule of that symmetry, let's create a U1 symmetry and a Z_2 symmetry and print their info:

* In python:

.. code-block:: python
    :linenos:


    sym_u1 = Symmetry.U1()
    sym_z2 = Symmetry.Zn(2)
    print(sym_u1)
    print(sym_z2)
    
* In c++:

.. code-block:: c++
    :linenos:

    Symmetry sym_u1 = Symmetry::U1();
    Symmetry sym_z2 = Symmetry::Zn(2);

    cout << sym_u1 << endl;
    cout << sym_z2 << endl;

Output >>

.. code-block:: text

    --------------------
    [Symmetry]
    type : Abelian, U1
    combine rule : Q1 + Q2
    reverse rule : Q*(-1) 
    --------------------

    --------------------
    [Symmetry]
    type : Abelian, Z(2)
    combine rule : (Q1 + Q2)%2
    reverse rule : Q*(-1) 
    --------------------


Create Bond with Qnums
*****************************

When system has some symmetry, the Bond can carry quantum numbers. To construct a Bond with symmetry and associate quantum numbers, the following API can be use:

.. py:function:: Bond(bd_type, qnums_list , degeneracies, sym_list)
     
    :param bondType bd_type: The type (direction) of the bond, this can ONLY be BD_KET--inward (BD_IN) or BD_BRA--outward (BD_OUT) when carry quantum numbers.
    :param list qnums_list: The quantum number list
    :param list degeneracies: The degeneracies(dimensions) of each qnums. 
    :param list sym_list: The list symmetries objects that defines  the type of each qnums.


    
The two arguments *qnums_list* and *degeneracies* can be combined into single one, for example:

* In python:

.. code-block:: python 
    :linenos:
    
    # This creates an KET (IN) Bond with quantum number 0,-4,-2,3 with degs 3,4,3,2 respectively.
    bd_sym_u1_a = Bond(BD_KET,[Qs(0)>>3,Qs(-4)>>4,Qs(-2)>>3,Qs(3)>>2],[Symmetry.U1()])

    # equivalent:
    bd_sym_u1_a = Bond(BD_IN,[Qs(0),Qs(-4),Qs(-2),Qs(3)],[3,4,3,2],[Symmetry.U1()])

    print(bd_sym_u1_a)

* In C++:

.. code-block:: c++
    :linenos:
    
    Bond bd_sym_u1_a = Bond(BD_KET,{Qs(0)>>3,Qs(-4)>>4,Qs(-2)>>3,Qs(3)>>2},{Symmetry::U1()});
    
    Bond bd_sym_u1_a = Bond(BD_IN,{Qs(0),Qs(-4),Qs(-2),Qs(3)},{0,4,3,2},{Symmetry::U1()});

    print(bd_sym_u1_a);

Output >>

.. code-block:: text

    Dim = 12 |type: KET>     
     U1::   +0  -4  -2  +3
    Deg>>    3   4   3   2


In some cases, we might want to include multiple symmetries in the system. For example: U1 x Z2, which can be achieve by adding additional quantum numbers inside *Qs()*


.. code-block:: python 
    :linenos:

    # This creates an KET (IN) Bond with U1xZ2 with quantum number (0,0),(-4,1),(-2,0),(3,1) with degs 3,4,3,2 respectively.
    bd_sym_u1z2_a = Bond(BD_KET,[Qs(0 ,0)>>3,\
                                 Qs(-4,1)>>4,\
                                 Qs(-2,0)>>3,\
                                 Qs(3 ,1)>>2],[Symmetry.U1(),Symmetry.Zn(2)])

    print(bd_sym_u1z2_a)


* In C++:

.. code-block:: c++
    :linenos:

    auto bd_sym_u1z2_a = Bond(BD_KET,{Qs(0 ,0)>>3,
                                     Qs(-4,1)>>4,
                                     Qs(-2,0)>>3,
                                     Qs(3 ,1)>>2},{Symmetry::U1(),Symmetry::Zn(2)});

    print(bd_sym_u1z2_a);
    

.. code-block:: text

    Dim = 12 |type: KET>     
     U1::   +0  -4  -2  +3
     Z2::   +0  +1  +0  +1
    Deg>>    3   4   3   2



Combine Bonds
*****************
    Now lets see how to combine two different Bonds. Let's create another U1 bond **bd_sym_u1_c**, and conbine it with **bd_sym_u1_a**:

* In python:

.. code-block:: python
    :linenos:

    bd_sym_u1_c = Bond(BD_KET,[Qs(-1)>>2,Qs(1)>>3,Qs(2)>>4,Qs(-2)>>5,Qs(0)>>6])
    print(bd_sym_u1_c)

    bd_sym_all = bd_sym_u1_a.combineBond(bd_sym_u1_c)
    print(bd_sym_all)


Output >>

.. code-block:: text

    Dim = 20 |type: KET>     
    U1::   -1  +1  +2  -2  +0
    Deg>>    2   3   4   5   6


    Dim = 240 |type: KET>     
     U1::   -6  -5  -4  -3  -2  -1  +0  +1  +2  +3  +4  +5
    Deg>>   20   8  39  18  49  15  30  19  16  12   6   8


Here we can observe the quantum numbers of **bd_sym_u1_a** combine with **bd_sym_u1_c** and generated 12 quantum numbers, respecting the combine rule (addition) of U1 symmetry.


.. Note::

    The Bonds need to be in the same direction to be combined. As physical interpretation, one cannot combine a ket state with a bra state!



.. Tips::
    
    using **combineBond_()** (with underscore) will modify the instance directly (as the general convention with underscore indicates inplace) 

combineBond by default will group any same quantum number together. Generally, the quantum number of merging two Bonds should be similar to Kron, and sometimes user might want to keep the order instead. In such scenario, one can set additioanll argument **is_grp = False**:


* In python:

.. code-block:: python
    :linenos:

    bd_sym_all = bd_sym_u1_a.combineBond(bd_sym_u1_c,is_grp=False)
    print(bd_sym_all)


.. code-block:: text
    

    Dim = 240 |type: KET>     
     U1::   -1  +1  +2  -2  +0  -5  -3  -2  -6  -4  -3  -1  +0  -4  -2  +2  +4  +5  +1  +3
    Deg>>    6   9  12  15  18   8  12  16  20  24   6   9  12  15  18   4   6   8  10  12




.. toctree::

