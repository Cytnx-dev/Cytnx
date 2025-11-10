Get/set UniTensor element
--------------------------

In this section, we discuss how to get an element directly from a UniTensor and how to set an element. Generally, elements can be accessed by first getting the corresponding block, and then accessing the correct element from that block. However, it is also possible to directly access an element from a UniTensor.

To get an element, one can call **UniTensor.at()**. It returns a *proxy* which contains a reference to the element. Furthermore, the proxy can be used to check whether an element corresponds to a valid block in a UniTensor with symmetries.


UniTensor without symmetries
*****************************

Accessing an element in a UniTensor without symmetries is straightforward by using *at*

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_elements_at_get.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_uniten_elements_at_get.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_elements_at_get.out
    :language: text

.. Note::

    Note that in Python, adding *.value* is necessary!


The proxy returned by **at** also serves as reference, so we can directly assign or modify the value:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_elements_at_set.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_uniten_elements_at_set.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_elements_at_set.out
    :language: text


UniTensor with symmetries
*****************************

When a UniTensor has block structure, not all possible elements correspond to a valid block. Invalid elements do not fulfill the symmetries. Therefore, these invalid elements should not be accessed.

In such cases, one can still use *at* and receive a proxy. The proxy can be used to check if the element is valid.

Let's consider the same example of a symmetric tensor as in the previous sections:

.. image:: image/u1_tdex.png
    :width: 500
    :align: center

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_elements_init_sym.py
    :language: python
    :linenos:

An existing element (here: at [0,0,0]) can be accessed as in the case without symmetries:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_elements_at_qidx.py
    :language: python
    :linenos:

* In C++:

.. code-block:: c++
    :linenos:

    print(Tsymm.at({0,0,0}));

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_elements_at_qidx.out
    :language: text


If we try to access an element that does not correspond to a valid block (for example at [0,0,1]), an error is thrown:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_elements_at_non_exist.py
    :language: python
    :linenos:

* In C++:

.. code-block:: c++
    :linenos:

    print(Tsymm.at({0,0,1}));

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_elements_at_non_exist.out
    :language: text

To avoid this error, we can check if the element is valid before accessing it. The proxy provides the method **exists()** for this purpose. For example, if we want to assign the value 8 to all valid elements with indices of the form [0,0,i], we can use:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_elements_exists.py
    :language: python
    :linenos:

* In C++:

.. code-block:: c++
    :linenos:

    for(auto i=0;i<2;i++){
        auto tmp = Tsymm.at({0,0,i});
        if(tmp.exists()):
            tmp = 8;
    }


This will set the element at [0,0,0] to 8 while ignoring the [0,0,1] element that does not exist.


.. toctree::
