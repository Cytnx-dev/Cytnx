Tagged UniTensor
-------------------

In this section we introduce **tagged UniTensor**, which is a UniTensor with *directional* bonds. Mathematically, a bond directing towards a tensor represents a vectorspace, while a bond directing away from a tensor represents the corresponding dual space. In physics, this is often represented as a "Ket" or "Bra" vector. We use the convention that "Ket" is represented by a bond directing towards the tensor, while "Bra" points away from the tensor.

If symmetric tensors are considered, the bonds carry quantum numbers and are always tagged. Therefore, tagged UniTensors can further be categorized into **non-symmetric** and **symmetric** (block form) tensors. The latter are covered in :ref:`UniTensor with Symmetries`.

Non-symmetric Tagged UniTensor
********************************

To create the non-symmetric, tagged UniTensor, we initialize it with **directional** bonds (BD_IN/BD_KET or BD_OUT/BD_BRA). Let's create UniTensors **Ta**, **Tb** and **Tc** in this way and investigate their properties:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_tagged_init.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_tagged_init.out
    :language: text

            
In this example, the UniTensors **Ta** and  **Tb**  are created to be in the **braket form (braket_form : True)**. This means that all bonds in the rowspace are Kets (inward) and all the bonds in the colspace are Bras(outward), which can be seen in the diagram as well. The property rowrank of a UniTensor defines these spaces: the first bonds are part of rowspace, and rowspace contains *rowrank* bonds.

Two bonds can only be contracted when one of them corresponds to a Ket state (living in a vector space) and the other is Bra state (living in the dual space). Therefore, **two bonds with conflicting directions cannot be contract with each other**. While this is certainly a constraint, this condition makes the implementation of tensor network algorithms less error-prone since incompatible bond directions will not be contracted.

To demonstrate this, we use :ref:`contract` and try:


.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_tagged_contract.py
    :language: python
    :linenos:

While the first contraction works well, the second one throws a RuntimeError stating "BRA-KET mismatch":

Output >> 

.. literalinclude:: ../../../code/python/outputs/guide_uniten_tagged_contract.out
    :language: text
    :lines: 1-3


.. toctree::
