Get/set UniTensor element
--------------------------

In this section, we discuss how to get an element directly from a UniTensor and how to set an element. Generally, elements can be accessed by first getting the corresponding block, and then accessing the correct element from that block. However, it is also possible to directly access an element from a UniTensor.

To get an element, one can call **UniTensor.at()**. It returns a *proxy* which contains a reference to the element. Furthermore, the proxy can be used to check whether an element corresponds to a valid block in a UniTensor with symmetries.


UniTensor without symmetries
*****************************

Accessing an element in a UniTensor without symmetries is straightforward by using *at*

* In Python:

.. code-block:: python
    :linenos:

    T = cytnx.UniTensor(cytnx.arange(9).reshape(3,3))
    print(T.at([0,2]).value)
   
* In C++:

.. code-block:: c++
    :linenos:

    auto T = cytnx::UniTensor(cytnx::arange(9).reshape(3,3));
    print(T.at({0,2}));

Output >> 

.. code-block:: text

    2.0

.. Note::

    Note that in Python, adding *.value* is necessary! 


The proxy returned by **at** also serves as reference, so we can directly assign or modify the value:

* In Python:

.. code-block:: python
    :linenos:
    
    T = cytnx.UniTensor(cytnx.arange(9).reshape(3,3))
    print(T.at([0,2]).value)
    T.at([0,2]).value = 7
    print(T.at([0,2]).value)

* In C++:

.. code-block:: c++
    :linenos:
    
    auto T = cytnx::UniTensor(cytnx::arange(9).reshape(3,3));
    print(T.at({0,2}));
    T.at({0,2}) = 7;
    print(T.at({0,2}));


Output >> 

.. code-block:: text

    2.0
    7.0



    
UniTensor with symmetries
*****************************

When a UniTensor has block structure, not all possible elements correspond to a valid block. Invalid elements do not fulfill the symmetries. Therefore, these invalid elements should not be accessed.

In such cases, one can still use *at* and receive a proxy. The proxy can be used to check if the element is valid.

Let's consider the following example:

.. image:: image/u1_tdex.png
    :width: 500
    :align: center

* In Python:

.. code-block:: python
    :linenos:
    
    bond_c = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],[cytnx.Symmetry.U1()])
    bond_d = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],[cytnx.Symmetry.U1()])
    bond_e = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(2)>>1, cytnx.Qs(0)>>2, cytnx.Qs(-2)>>1],[cytnx.Symmetry.U1()])
    Td = cytnx.UniTensor([bond_c, bond_d, bond_e]);


An existing element (here: at [0,0,0]) can be accessed as in the case without symmetries:

* In Python:

.. code-block:: python
    :linenos:

    print(Td.at([0,0,0]).value)


* In C++:

.. code-block:: c++
    :linenos:

    print(Td.at({0,0,0}));
        
        

Output>>

.. code-block:: text
    
    0.0


If we try to access an element that does not correspond to a valid block (for example at [0,0,1]), an error is thrown:

* In Python:

.. code-block:: python
    :linenos:

    print(Td.at([0,0,1]).value)


* In C++:

.. code-block:: c++
    :linenos:

    print(Td.at({0,0,1}));

Output>>

.. code-block:: text
    
    ValueError: [ERROR] trying access an element that is not exists!, using T.if_exists = sth or checking with T.exists() to verify before access element!


To avoid this error, we can check if the element is valid before accessing it. The proxy provides the method **exists()** for this purpose. For example, if we want to assign the value 8 to all valid elements with indices of the form [0,0,i], we can use:

* In Python:

.. code-block:: python
    :linenos:

    for i in [0,1]:
        tmp = Td.at([0,0,i])
        if(tmp.exists()):
            tmp.value = 8.

* In C++:

.. code-block:: c++
    :linenos:

    for(auto i=0;i<2;i++){
        auto tmp = Td.at({0,0,i});
        if(tmp.exists()):
            tmp = 8;
    }


This will set the element at [0,0,0] to 8 while ignoring the [0,0,1] element that does not exist. 




.. toctree::
