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

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_bond_create.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_bond_create.out
    :language: text


In some scenarios, one may want to change the direction of the bond, namely from BD_BRA
to BD_KET or the opposite, one can then use .redirect() method:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_bond_redirect.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_bond_redirect.out
    :language: text


Symmetry object
**********************

Symmetries play an important role in physical simulations. Tensors and bonds can be defined in a way that preserves the symmetries. This helps to reduce the numerical costs, can increase precision and it allows to do calculations restricted to specific symmetry sectors.

In Cytnx, the symmetry type is defined by a Symmetry object. It contains the name, type, combine rule and the reverse rule of that symmetry. The combine rule contains the information how two quantum numbers are combined to a new quantum number. Let us create Symmetry objects for a *U1* and a *Z_2* symmetry and print their info:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_bond_symobj.py
    :language: python
    :linenos:

* In C++:
.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_uniten_bond_symobj.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_bond_symobj.out
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

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_bond_sym_bond.py
    :language: python
    :linenos:

* In C++:
.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_uniten_bond_sym_bond.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_bond_sym_bond.out

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_bond_multi_sym_bond.py
    :language: python
    :linenos:

* In C++:
.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_uniten_bond_multi_sym_bond.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_bond_multi_sym_bond.out
    :language: text

Combining Bonds
*****************

In typical algorithms, two bonds often get combined to one bond. This can be done with Bonds involving Symmetries as well. The quantum numbers are merged according to the combine rules.

As an example, let us create another *U1* Bond **bd_sym_u1_c** and combine it with **bd_sym_u1_a**:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_bond_combine.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_bond_combine.out
    :language: text


Here, we can observe the quantum numbers of **bd_sym_u1_a** combine with **bd_sym_u1_c** and generate 12 quantum numbers, respecting the combine rule (addition) of the *U1* symmetry.


.. note::

    The Bonds need to be in the same direction to be combined. As a physical interpretation, one cannot combine a ket state with a bra state.

.. warning::

    When no symmetry argument is given in the creation of a Bond with quantum numbers, *U1* is assumed by default as the symmetry group.


.. tip::

    Using **combineBond_()** (with underscore) will modify the instance directly (as the general convention with underscore indicates inplace).

By default, *combineBond* will group any quantum numbers of the same type together. If one wants to keep the order instead, similarly to *Kron*, one can set the additional argument **is_grp = False**:


* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_bond_combine_no_grp.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_bond_combine_no_grp.out
    :language: text

.. warning::

    This is not efficient since duplicate quantum number can occur. A warning will be thrown when is_grp=False is used.



.. toctree::
