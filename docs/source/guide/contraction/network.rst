Network
========================

The **Network** object provides a skeleton of a Tensor network diagram. Users can write a *Network file*, which serves as the blue print of a TN structure. With a *Network* defined in such a way, multiple Tensors can be contracted at once. The contraction order can either be specified or automatically optimized by Cytnx. Furthermore, the tensor network can be printed to check the implementation.

*Network* is useful when we have to contract different tensors with the same connectivity many times. We can define the *Network* once and reuse it for several contractions. The contraction order can be optimized once after the first initialization with tensors. In proceeding steps this optimized order can be reused.

Network from .net file
--------------------------

We take a typical contraction in a corner transfer matrix algorithm as an example. The TN diagram is given by:

.. image:: image/ctm.png
    :width: 300
    :align: center

We implement the diagram as a .net file to represent the contraction task:

* ctm.net:

.. literalinclude:: ../../../code/ctm.net
    :language: text


Note that:

1. The labels above correspond to the diagram you draw, not the label attribute of UniTensor objects. Both label conventions can, but do not have to be the same.
   
2. Labels should be separated by ' , '.
   
3. In TOUT, a ' ; ' separates the labels in rowspace and colspace, note that it is optional, if there is no ' ; ' specified, all label will be put in colspace.
   
4. TOUT specifies the output configuration, in this case we leave it blank since the result will be a scalar.
   
5. ORDER is optional and used to specify the contraction order manually.


Launch
--------------------------

To perform the contraction and get the outcome, we use the **.Launch()**:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_contraction_network_launch.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_contraction_network_launch.cpp
    :language: c++
    :linenos:



Network from string
--------------------------
Alternatively, we can implement the contraction directly in the program with FromString(): 

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_contraction_network_FromString.py
    :language: python
    :linenos:

This approach can be convenient if you do not want to maintain the .net files.


.. toctree::


Put UniTensors
--------------------------
We use the .net file to create a Network. Then, we can load instances of UniTensors:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_contraction_network_PutUniTensor.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_contraction_network_PutUniTensor.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_contraction_network_PutUniTensor.out
    :language: text

.. Note::
        The indices of the UniTensors to be put into the Network need to be ordered according to the indices in the .net file. Otherwise, the index order can be defined in PutTensor explicitly, see :ref:`PutUniTensor according to label ordering` below.

PutUniTensor according to label ordering
------------------------------------------

When we put a UniTensor into a Network, we can also specify its leg order according to a label ordering, this interface turns out to be convinient since users don't need to memorize look up the index of s desired leg. To be more specific, consider a example, we grab two three leg tensors **A1** and **A2**, they both have one leg that spans the physical space and the other two legs describe the virtual space (such tensors are often appearing as the building block tensors of matrix product state), we create the tensors and set the corresponding lebels as following:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_contraction_network_label_ord-1.py
    :language: python
    :linenos:

The legs of these tensors are arranged such that the first leg is the physical leg (with dimension 2 for spin-half case for example) and the other two legs are the virtual ones (with dimension 8).

Now suppose somehow we want to contract these two tensors by its physical legs, we create the following Network:

Note that in this Network it is the second leg of the two tensors to be contracted, which will not be consistent since **A1** and **A2** are created such that their physical leg is the first one, while we can do the following:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_contraction_network_label_ord-2.py
    :language: python
    :linenos:

Note that in this Network the second leg of the two tensors are to be contracted. This is not consistent to the definition of **A1** and **A2** which are created such that their physical leg is the first one. We can call `PutUniTensor` and specify the labels though:


* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_contraction_network_label_ord-3.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_contraction_network_label_ord-3.out
    :language: text

So when we do the PutUniTensor() we add the third argument which is a labels ordering, what this function will do is nothing but permute the tensor legs according to this label ordering before putting them into the Network.

So when calling `PutUniTensor()` we add the third argument which is a labels ordering. This will permute the tensor legs according to the given label ordering before putting them into the Network.


Set the contraction order
--------------------------

To set or find the optimal contraction order of our tensor network, we provide **.setOrder(optimal, contract_order)** function, by passing true for the first argument, the Network will find an optimal contraction order for us and store it: 

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_contraction_network_Optimal.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_contraction_network_Optimal.cpp
    :language: c++
    :linenos:

.. Note::
    
    Although Network does cache the optimal contraction order once it is found, the optimal order finding rountine will still be executed everytime we call the .setOrder(optimal= true), it is suggested that one store the optimal order and specify it manually in some situaitions to prevent the overhead of re-finding optimal order.


We can also pass the string specifying our desired contraction order in the second arguement:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_contraction_network_setOrder.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_contraction_network_setOrder.cpp
    :language: c++
    :linenos:

.. Note::
    
    By default the **optimal = False** , so for python case we can ignore the optimal argument and just pass the order. Note that if one pass **optimal = True** while specifying the order at the same time, Network will find and use the optimal order.


To inspect the current contraction order stored in the Network, we can use **.getOrder()** which returns a contraction order string:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_contraction_network_getOrder.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_contraction_network_getOrder.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_contraction_network_getOrder.out
    :language: text


