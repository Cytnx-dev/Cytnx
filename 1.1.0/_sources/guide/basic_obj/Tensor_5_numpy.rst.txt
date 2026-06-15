To/From numpy.array
----------------------
Cytnx also provides conversion from and to numpy.array in the Python API.

* To convert from Cytnx Tensor to numpy array, use **Tensor.numpy()**

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Tensor_5_numpy_cytnx2numpy.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Tensor_5_numpy_cytnx2numpy.out
    :language: text

* To convert from numpy array to Cytnx Tensor, use **cytnx.from_numpy()**

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Tensor_5_numpy_numpy2cytnx.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Tensor_5_numpy_numpy2cytnx.out
    :language: text


.. toctree::
