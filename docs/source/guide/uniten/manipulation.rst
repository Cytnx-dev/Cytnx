Manipulating a UniTensor
-------------------------

After having introduced the initialization and structure of the three UniTensor types (un-tagged, tagged, and tagged with symmetries), we show basic functionalities to manipulate UniTensors.

Permutation, reshaping and arithmetic operations are accessed similarly to **Tensor** objects as introduced before, with slight modifications for symmetric UniTensors.

Permute
************************************

The bond order can be changed with *permute* for all kinds of UniTensors. The order can either be defined by the index order as for the permute method of a *Tensor*, or by specifying the label order after the permutation.

For example, we permute the indices of the symmetric tensor that we introduced before:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_manipulation_permute.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_manipulation_permute.out
    :language: text


We did the same permutation in two ways in this example, once using indices, once using labels. The first index of the permuted tensor corresponds to the last index of the original tensor (original index 2, label "f"), the second new index to the first old index (old index 0, label "d") and the last new bond has the old index 1 and label "e".

Reshape
************************************

Untagged UniTensors can be reshaped just like normal Tensors.

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_manipulation_reshape.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_manipulation_reshape.out
    :language: text

.. Note::

    A tagged UniTensor can not be reshaped. This includes symmetric UniTensors as well.

Combine bonds
************************************

Tagged UniTensors, including symmetric UniTensors, cannot be reshaped. The reason for this is that the bonds to be combined or split include the direction and quantum number information.
A reshape is therefore replaced by the fusion or splitting of indices, which takes into account the transformation of the quantum numbers for the given symmetries. For this, Cytnx provides the combineBonds API for tagged UniTensor as an alternative to the usual reshape.
Note that there is currently no API for splitting bonds, since the way to split the quantum basis is ambiguous.
The method to combine bonds can be used as follows:


.. py:function:: UniTensor.combineBonds(indicators, force)

    :param list indicators: A list of **integer** indicating the indices of bonds to be combined. If a list of **string** is passed the bonds with those string labels will be combined.
    :param bool force: If set to **True** the bonds will be combined regardless the direction or type of the bonds, otherwise the bond types will be checked. The default is **False**.


The use is demonstrated in this example:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_manipulation_combine.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_manipulation_combine.out
    :language: text


Arithmetic
************************************


Arithmetic operations for un-tagged UniTensors can be done in the exact same way as for Tensors, see :ref:`Tensor arithmetic`. The supported arithmetic operations and further linear algebra functions are listed in :ref:`Linear algebra`.

Rowrank
*********

Another property of a UniTensor that we need to control is its rowrank. It defines how the legs of the a UniTensor are split into two halves, one part belonging to the rowspace and the other to the column space. A UniTensor can then be seen as a linear operator between these two spaces, or as a matrix. The matrix form corresponds to combining the first *rowrank* indices to a single (row-)index and the remaining indices to a second (column-)index.

Most of the linear algebra algorithms assume this matrix form as an input. We thus use rowrank to specify how to interpret the input UniTensor as a matrix. This specification makes it easy to use linear algebra operations on UniTensors, even if they have more than two indices.

The rowrank can either be specified when initializing the UniTensor, or the **.set_rowrank()** method can be used to modify the rowrank of a UniTensor:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_manipulation_rowrank.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_manipulation_rowrank.out
    :language: text


How linear algebra functions such as the **singular value decomposition (SVD)** make use of the rowrank is described in chapter :ref:`Tensor decomposition`.


Transpose
**********************

One common operation that is sensitive to the **rowrank** is transposing a tensor. This is possible for a UniTensor by the method **.Transpose()** (or the in-placed method **.Transpose_()**).
We first show the behavior for a **non-tagged** UniTensor:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_manipulation_Transpose.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_manipulation_Transpose.out
    :language: text


We see that .Transpose() swap the legs in the row space and the column space, with the *rowrank* itself also being modified.

Next we consider the transposition of a tagged UniTensor:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_manipulation_Transpose_tagged.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_manipulation_Transpose_tagged.out
    :language: text


In addition to exchanging the roles of row- and column-space as before, **the direction of each bond is inverted**.

.. Note::

    1. The method **Transpose_()** works similarly, but changes the UniTensor directly instead of generating a new UniTensor.
    2. For a :ref:`Fermionic UniTensor`, the order of the indices after transposing the tensor is inverted.


Dagger
**********************

The methods **.Dagger()** and **.Dagger_()** correspond to the conjugate-transpose of a tensor, similar to applying .Conj() and .Transpose(). See the previous section :ref:`Transpose` for the behavior.
