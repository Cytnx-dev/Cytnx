Contract
=============
Two UniTensors can be contracted with the function **Contract()**. Indices with the same labels on two of the input tensors are summed over. The contraction order can additionally be specified.

Contracting two UniTensors
------------------------------------

The function **cytnx.Contract()** contracts all common labels of two UniTensors. For example:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_contraction_contract_Contract.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_contraction_contract_Contract.out
    :language: text


Here we see that the labels **j** and **l** appear on both input tensors. Thus, they are contracted. Note that the bond dimensions of the contracted tensors must agree on both tensors.

In order to define which indices shall be contracted without changing the labels on the initial tensors, Cyntx provides the method **.relabel()**. It allows to set common labels on the indices to be contracted and distinct labels on the others. Also, the labels on the resulting tensor can be defined this way. See :ref:`Changing labels` for further details. Suppose that we only want to contract the index *j* in the previous example, but not sum over *l*. We can use **.relabel()** for this task:


* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_contraction_contract_relabel.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_contraction_contract_relabel.out
    :language: text


The function **.relabel()** creates a copy of the initial UniTensor and changes the labels, while keeping the labels on the initial tensor unchanged. The actual data is shared between the old and new tensor, only the metadata is independent.

Contracting multiple UniTensors
------------------------------------
The function **Contract** also allows one to contract multiple UniTensors.

The first argument in this case is **TNs**, which is a list containing all UniTensors to be contracted. Contract also provides the argument **order** to specify a desired contraction order, or the **optimal** option to use an automatically optimized contraction order.

Consider the following contraction task consisting of UniTensors **A1**, **A2** and **M**:

.. image:: image/contracts.png
    :width: 300
    :align: center

This corresponds to the Python program:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_contraction_contract_Contracts.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_contraction_contract_Contracts.out
    :language: text

Note that the UniTensors' names have to be specified for an explicitly given contraction order. In this case, we specified them by the method set_name. The order *(M,(A1,A2))* indicates that first all common indices of *A1* and *A2* are contracted, then all common indices of the resulting tensor and *M*.

.. Note::
    All tensors contracted with `Contract()` need to have unique tensor names. Use `UniTensor.set_name()` to specify the name of a tensor.

.. warning::

    The function *Contracts()* is deprecated. Use *Contract()* instead with a list of input tensors.
