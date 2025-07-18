Changing labels
------------------
We can set and change the labels of the Bonds in a UniTensor as desired. This is particularly helpful for contractions with *cytnx.Contract()* and *cytnx.Contracts()*. As will be explained in :ref:`Contract(s)`, these functions contract bonds with the same name on different UniTensors. Therefore, we might need to change the labels for some bond(s) to initiate the correct tensor contraction.

To change the label associated to a certain leg of a UniTensor, one can use:

.. py:function:: UniTensor.relabel_(index, new_label)

    :param [int] index: the index (order) of the bond in the UniTensor
    :param [string] new_label: the new label that you want to change to


Alternatively, if we don't know the index of the target bond in the current order, we can also specify the old label:

.. py:function:: UniTensor.relabel_(old_label, new_label)

    :param [string] old_label: the current label of the bond
    :param [string] new_label: the new label that you want to change to


If we wish to change the labels of all legs, we can use:

.. py:function:: UniTensor.relabels_( new_labels)

    :param List[string] new_labels: a list of new labels

or

.. py:function:: UniTensor.relabels_(old_labels, new_labels)

    :param List[string] old_labels: a list of current labels
    :param List[string] new_labels: a list of the corresponding new labels

For example:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_labels_relabel_.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_labels_relabel_.out
    :language: text


.. note::

    One cannot have duplicate labels *within* the same UniTensor!


.. warning::

    The previously provided method set_label(s) is deprecated and should be replaced by relabel(s)_.


Creating UniTensors with different labels that share data
*********************************************************

In some scenarios, especially in contractions with *cytnx.Contract()* and *cytnx.Contracts()*, we want to create a UniTensor with changed labels. However, we might not want to modify the original tensor. Creating a copy of the tensor data is also not desired, since it would double the memory usage. In such a case one can use the function **relabel(s)** without underscore. This returns a new UniTensor with different meta (in this case  only the labels are changed), but the actual memory block(s) are still referring to the old ones. The arguments of **relabel(s)** are similar to **relabel(s)_**, see above. For example:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_labels_relabel.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_labels_relabel.out
    :language: text


.. toctree::
