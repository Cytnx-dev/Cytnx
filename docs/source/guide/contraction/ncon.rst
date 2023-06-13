ncon
=============
The **ncon** is another useful function to reduce users' programming effort required to implement a tensor network contraction, which is originally proposed for MATLAB :cite:`pfeifer2015ncon`.

To use ncon, we first make a labelled diagram of the desired network contraction such that:
Each internal index (index to be contracted) is labelled with a unique positive integer (typically sequential integers starting from 1, although this is not necessary).

External indices of the diagram (if there are any) are labelled with sequential negative integers [-1,-2,-3,…] which denote the desired index order on the final tensor (with -1 as the first index, -2 as the second etc).

Following this, the **ncon** routine is called as follows,

.. py:function:: OutputTensor = ncon(tensor_list_in, connect_list_in, cont_order)
     
    :param list tensor_list_in: 1D array containing the tensors comprising the network
    :param list connect_list_in: 1D array of vectors, where the kth element is a vector of the integer labels from the diagram on the kth tensor from tensor_list_in (ordered following the corresponding index order on this tensor).
    :param list cont_order: a vector containing the positive integer labels from the diagram, used to specify order in which **ncon** contracts the indices. Note that cont_order is an optional input that can be omitted if desired, in which case ncon will contract in ascending order of index lab.

For example, we want to contract the following tensor network (again) consists of tensors **A1**, **A2** and **M**:

.. image:: image/ncon.png
    :width: 300
    :align: center

In the figure we labelled the internal leg using the unique positive numbers and extermal legs the negative ones, translate this figure
to the ncon function calling we have:

* In Python:

.. code-block:: python
    :linenos:

    # Creating A1, A2, M
    A1 = cytnx.UniTensor(cytnx.ones([2,8,8]))
    A2 = cytnx.UniTensor(cytnx.ones([2,8,8]))
    M = cytnx.UniTensor(cytnx.ones([2,2,4,4]))

    # Calling ncon
    res = cytnx.ncon([A1,M,A2],[[1,-1,-2],[1,2,-3,-4],[2,-5,-6]])

We see that **ncon** accomplish the similar thing as **Contracts**, just now the labeling of the UniTensors in the network 
is incorporated into the function argument, thus make the code more compact.

.. bibliography:: ref.ncon.bib
    :cited: