Bond
=======
A **Bond** is an object that represents the legs or indices of a tensor. It carries information such as the direction, dimension and quantum numbers (if symmetries given). 

There are in general two types of Bonds: **directional** and **undirectional**, depending on whether the bond has a direction (pointing inward or outward with respect to the tensor) or not. The inward Bond is also defined as **Ket**/**In** type, while the outward Bond is defined as **Bra**/**Out** type as in the *Braket* notation in the quantum mechanics: 

.. image:: image/bond.png
    :width: 400
    :align: center

The API for constructing a simple Bond (with or without direction) is:

.. py:function:: Bond(dim, bd_type)
     
    :param int dim: The dimension of the bond.
    :param bondType bd_type: The type (direction) of the bond, can be BD_REG--undirectional, BD_KET--inward (same as BD_IN), BD.BRA--outward (same as BD_OUT)


A Bond object (without any symmetry) can thus be created as follows:

* In Python:

.. code-block:: python
    :linenos:

    from cytnx import Bond
    # This creates an in-going Bond with dimension 10.
    bond_1 = Bond(10, BD_IN)
    print(bond_1)
    # If one doesn't specify the Bond type, the default bond type will be
    regular or undirectional.
    bond_2 = Bond(10)
    print(bond_2)

Output >>

.. code-block:: text

    Dim = 10 |type: KET>
    Dim = 10 |type: REGULAR>

In some scenarios, one may want to change the direction of the bond, namely from BD_BRA
to BD_KET or the opposite, one can then use .redirect() method:

* In Python:

.. code-block:: python
    :linenos:

    from cytnx import Bond
    bond_1 = Bond(10, BD_IN)
    bond_2 = bond_1.redirect()
    print(bond_1)
    print(bond_2)

Output >>

.. code-block:: text

    Dim = 10 |type: KET>
    Dim = 10 |type: <BRA


Symmetry object
**********************

Symmetries play an important role in physical simulations. Tensors and bonds can be defined in a way that preserves the symmetries. This helps to reduce the numerical costs, can increase precision and it allows to do calculations restricted to specific symmetry sectors.

In Cytnx, the symmetry type is defined by a Symmetry object. It contains the name, type, combine rule and the reverse rule of that symmetry. The combine rule contains the information how two quantum numbers are combined to a new quantum number. Let us create Symmetry objects for a *U1* and a *Z_2* symmetry and print their info:

* In Python:

.. code-block:: python
    :linenos:


    sym_u1 = cytnx.Symmetry.U1()
    sym_z2 = cytnx.Symmetry.Zn(2)
    print(sym_u1)
    print(sym_z2)
    
* In C++:

.. literalinclude:: ../../../code/cplusplus/guide_codes/7_4_1_ex1.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/cplusplus/outputs/7_4_1_ex1.out
    :language: text


Creating Bonds with quantum numbers
************************************

In order to implement symmetries on the level of tensors, we assign a quantum number to each value of an index. The quantum numbers can have a degeneracy, such that several values of an index correspond to the same quantum number.

To construct a Bond with symmetries and associate quantum numbers, the following API can be used:

.. py:function:: Bond(bd_type, qnums_list , degeneracies, sym_list)
     
    :param bondType bd_type: type (direction) of the bond, this can ONLY be BD_KET--inward (BD_IN) or BD_BRA--outward (BD_OUT) when quantum numbers are used 
    :param list qnums_list: quantum number list 
    :param list degeneracies: degeneracies (dimensions) of the qnums 
    :param list sym_list: list of Symmetry objects that define the symmetry of each qnum


    
The two arguments *qnums_list* and *degeneracies* can be combined into a single argument by using the following API:

.. py:function:: Bond(bd_type, qnums_degeneracy_pair, sym_list)

    :param bondType bd_type: type (direction) of the bond, this can ONLY be BD_KET--inward (BD_IN) or BD_BRA--outward (BD_OUT) when quantum numbers are used
    :param list qnums_degeneracy_pair_list: list of pairs of quantum numbers and degeneracies, which can be constructed with the helper class *Qs*
    :param list sym_list: list of Symmetry objects that define the symmetry of each qnum

*Qs* is a helper class that can be used to construct a pair of quantum number list and degeneracy.

For example:

* In Python:

.. code-block:: python 
    :linenos:
    
    # This creates an KET (IN) Bond with quantum number 0,-4,-2,3 with degs 3,4,3,2 respectively.
    bd_sym_u1_a = cytnx.Bond(cytnx.BD_KET,\
                            [cytnx.Qs(0)>>3,cytnx.Qs(-4)>>4,cytnx.Qs(-2)>>3,cytnx.Qs(3)>>2],\
                            [cytnx.Symmetry.U1()])

    # equivalent:
    bd_sym_u1_a = cytnx.Bond(cytnx.BD_IN,\
                            [cytnx.Qs(0),cytnx.Qs(-4),cytnx.Qs(-2),cytnx.Qs(3)],\
                            [3,4,3,2],[cytnx.Symmetry.U1()])

    print(bd_sym_u1_a)

* In C++:

.. literalinclude:: ../../../code/cplusplus/guide_codes/7_4_2_ex1.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/cplusplus/outputs/7_4_2_ex1.out
    :language: text

If several symmetries are present, this can be achieved by giving several quantum numbers inside *Qs()*. Let us consider a *U1 x Z2* symmetry for example:

* In Python:

.. code-block:: python 
    :linenos:

    # This creates a KET (IN) Bond with U1xZ2 symmetry
    # and quantum numbers (0,0),(-4,1),(-2,0),(3,1) with degs 3,4,3,2 respectively.
    bd_sym_u1z2_a = cytnx.Bond(cytnx.BD_KET,\
                               [cytnx.Qs(0 ,0)>>3,\
                                cytnx.Qs(-4,1)>>4,\
                                cytnx.Qs(-2,0)>>3,\
                                cytnx.Qs(3 ,1)>>2],\
                               [cytnx.Symmetry.U1(),cytnx.Symmetry.Zn(2)])

    print(bd_sym_u1z2_a)


* In C++:


.. code-block:: c++
    :linenos:

    auto bd_sym_u1z2_a = cytnx::Bond(cytnx::BD_KET,
                                     {cytnx::Qs(0 ,0)>>3,
                                      cytnx::Qs(-4,1)>>4,
                                      cytnx::Qs(-2,0)>>3,
                                      cytnx::Qs(3 ,1)>>2},
                                     {cytnx::Symmetry::U1(),cytnx::Symmetry::Zn(2)});

    print(bd_sym_u1z2_a);
    

Output >>

.. code-block:: text

    Dim = 12 |type: KET>     
     U1::   +0  -4  -2  +3
     Z2::   +0  +1  +0  +1
    Deg>>    3   4   3   2



Combining Bonds
*****************

In typical algorithms, two bonds often get combined to one bond. This can be done with Bonds involving Symmetries as well. The quantum numbers are merged according to the combine rules.

As an example, let us create another *U1* Bond **bd_sym_u1_c** and combine it with **bd_sym_u1_a**:

* In Python:

.. code-block:: python
    :linenos:

    bd_sym_u1_c = cytnx.Bond(cytnx.BD_KET,\
                    [cytnx.Qs(-1)>>2,cytnx.Qs(1)>>3,cytnx.Qs(2)>>4,cytnx.Qs(-2)>>5,cytnx.Qs(0)>>6])
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


Here, we can observe the quantum numbers of **bd_sym_u1_a** combine with **bd_sym_u1_c** and generate 12 quantum numbers, respecting the combine rule (addition) of the *U1* symmetry.


.. note::

    The Bonds need to be in the same direction to be combined. As a physical interpretation, one cannot combine a ket state with a bra state.

.. warning::

    When no symmetry argument is given in the creation of a Bond with quantum numbers, *U1* is assumed by default as the symmetry group. 



.. tip::
    
    Using **combineBond_()** (with underscore) will modify the instance directly (as the general convention with underscore indicates inplace). 

By default, *combineBond* will group any quantum numbers of the same type together. If one wants to keep the order instead, similarly to *Kron*, one can set the additional argument **is_grp = False**:


* In Python:

.. code-block:: Python
    :linenos:

    bd_sym_all = bd_sym_u1_a.combineBond(bd_sym_u1_c,is_grp=False)
    print(bd_sym_all)


Output >>

.. code-block:: text
    

    Dim = 240 |type: KET>     
     U1::   -1  +1  +2  -2  +0  -5  -3  -2  -6  -4  -3  -1  +0  -4  -2  +2  +4  +5  +1  +3
    Deg>>    6   9  12  15  18   8  12  16  20  24   6   9  12  15  18   4   6   8  10  12

.. warning::

    This is not efficient since duplicate quantum number can occur. A warning will be thrown when is_grp=False is used.



.. toctree::

