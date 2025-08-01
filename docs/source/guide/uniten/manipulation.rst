Manipulate UniTensor
--------------------

After having introduced the initialization and structure of the three UniTensor types (un-tagged, tagged and tagged with symmetries),
we show the basic functionalities to manipulate UniTensors.

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

The tagged UniTensors include symmetric UniTensors cannot be reshaped, since the bonds to be combinded or splited now includes the direction and quantum number infomation,
the reshape process involves the fusion or split of the qunatum basis, we provide combindBonds API for the tagged UniTensor as an alternative to the usual reshape function.
Note that currently there is no API for splitting a bond, since the way to split the quantum basis will be ambiguous.
Let's see the complete function usage for combining bonds:


.. py:function:: UniTensor.combineBonds(indicators, force)

    :param list indicators: A list of **integer** indicating the indices of bonds to be combined. If a list of **string** is passed the bonds with those string labels will be combined.
    :param bool force: If set to **True** the bonds will be combined regardless the direction or type of the bonds, otherwise the bond types will be checked. The default is **False**.


Consider a specific example:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_manipulation_combine.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_manipulation_combine.out
    :language: text


Arithmetic
************************************


Arithmetic operations for un-tagged UniTensors can be done exactly the same as with Tensors, see :ref:`Tensor arithmetic`. The supported arithmetic operations and further linear algebra functions are listed in :ref:`Linear algebra`.

Rowrank
*********

Another property that we may want to maintain in UniTensor is its rowrank. It tells us how the legs of the a UniTensor are split into two halves, one part belongs to the rowspace and the other to the column space. A UniTensor can then be seen as a linear operator between these two spaces, or as a matrix. The matrix results in having the first *rowrank* indices combined to the first (row-)index and the other indices combined to the second (column-)index.

Most of the linear algebra algorithms take a matrix as an input. We thus use rowrank to specify how to cast the input UniTensor into a matrix. In Cytnx, this specification makes it easy to use linear algebra operations on UniTensors.

The rowrank can be specified when initializing the UniTenosr, one can also use **.set_rowrank()** to modify the rowrank of a UniTensor:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_manipulation_rowrank.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_manipulation_rowrank.out
    :language: text


We leave the examples of linalg algebra operations incoporating the rowrank concept such as **singular value decomposition (SVD)** to the chapter :ref:`linalg extension`.


Transpose
**********************

One common operation that is sensitive to the **rowrank** of a UniTensor is the tranpose, one can transpose a UniTensor using **.Transpose()** (or the in-placed method **.Transpose_()**), let's see the behavior of this operation, first consider the transpose of a **non-tagged** UniTensor:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_manipulation_Transpose.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_manipulation_Transpose.out
    :language: text


We see that .Transpose() swap the legs in the row space and column space, also the *rowrank* itself is modified.

Next we consider the tranposition of a tagged UniTensor:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_manipulation_Transpose_tagged.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_manipulation_Transpose_tagged.out
    :language: text


We see that for the tagged UniTensor the rowrank (and the row/column space the legs belong to) is not changed, instead the .Transpose() **inverted the direction of each bond**.

.. Note::

    The operation **.Dagger()** (which is the transposition plus a conjugation) shows same behavior as transpose discussed above.
